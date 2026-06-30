# Running the ThinkBooster service locally

Stand up the ThinkBooster service on your own machine and call it as an
OpenAI-compatible API endpoint. The scaling **strategy** and **scorer** live in the URL,
so any OpenAI client can drive test-time scaling by pointing `base_url` at the service —
no other code change. This guide covers building, running, and calling the service. For
the full request/response reference, see [`api_guide.md`](api_guide.md).

## What you get

The service exposes the OpenAI `/v1/chat/completions` schema, with the strategy and
scorer as URL path segments:

```
<THINKBOOSTER_ENDPOINT>/v1/<strategy>/<scorer>/chat/completions
```

| Path segment | Valid values |
|---|---|
| `<strategy>` | `self_consistency`, `offline_bon`, `online_bon`, `beam_search` |
| `<scorer>` (optional) | `entropy`, `perplexity`, `sequence_prob`, `prm` |

`self_consistency` runs against a remote model (OpenRouter or any OpenAI-compatible API)
and needs no GPU. The other strategies score reasoning steps: the uncertainty scorers
(`entropy`, `perplexity`, `sequence_prob`) are read from the generator's
log-probabilities at near-zero cost, while `prm` runs a separate reward model that needs
a GPU.

## 1. Prerequisites

| Need | When |
|---|---|
| Python 3.10+ and `git` | always |
| An OpenRouter (or other OpenAI-compatible) API key | for remote generation, including `self_consistency` |
| 1× NVIDIA GPU (≥24 GB) + driver | only for local generation or the `prm` scorer |
| Docker + `nvidia-container-toolkit` | only for the Docker path with a GPU |

## 2. Get the code and install

```bash
git clone https://github.com/IINemo/thinkbooster.git
cd thinkbooster

pip install -e ".[service]"

# lm-polygraph powers the uncertainty scorers and is a separate, gitignored repo.
# Clone it into the repo root (needed for the scorers and the Docker build):
git clone https://github.com/IINemo/lm-polygraph.git

# Service config; set your keys here (each request can also carry its own key).
cp service_app/.env.example .env
```

## 3. Run the service

### Bare metal (simplest)

```bash
# API-only: self_consistency / remote generation, no GPU needed
export OPENROUTER_API_KEY=sk-or-...    # fallback key, used when a request omits its own
export PORT=8001
python service_app/main.py
# -> http://localhost:8001   (interactive docs at /docs)
```

Keep it alive across disconnects with `tmux` (`tmux new -t thinkbooster`, detach with
`Ctrl-b d`). Logs are written under `logs/<date>/<time>/service.log`. To use a local
generation model or the PRM scorer, set the GPU environment variables first — see
[§6](#6-using-a-gpu-local-generation-or-the-prm-scorer).

### Docker

```bash
docker compose up --build -d
curl http://localhost:8001/health      # {"status":"healthy", ...}
```

The committed compose file is API-only. To attach a GPU (for local generation or the PRM
scorer), add an override file — see [§6](#6-using-a-gpu-local-generation-or-the-prm-scorer).

## 4. Call it as an endpoint

Point any OpenAI client at the service and put the strategy (and scorer) in `base_url`.

### Self-consistency (no GPU, remote generation)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1/self_consistency",
    api_key="<YOUR_OPENROUTER_KEY>",
)
response = client.chat.completions.create(
    model="deepseek/deepseek-r1",
    messages=[
        {"role": "system", "content": "Think step by step. Put your final answer in \\boxed{}."},
        {"role": "user", "content": "How many positive integers below 1000 are divisible by 3 but not 7?"},
    ],
    extra_body={"num_paths": 8, "max_tokens": 4096},
)
print(response.choices[0].message.content)

meta = response.model_dump()["choices"][0]["tts_metadata"]
print("answer:", meta["selected_answer"], "| agreement:", meta["consensus_score"])
```

### Beam search or best-of-N with a scorer

Change the strategy and scorer by changing the URL. The uncertainty scorers need no extra
model; `prm` needs the GPU setup in [§6](#6-using-a-gpu-local-generation-or-the-prm-scorer).

```python
client = OpenAI(
    base_url="http://localhost:8001/v1/beam_search/entropy",
    api_key="<YOUR_OPENROUTER_KEY>",
)
response = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B",
    messages=[
        {"role": "system", "content": "Think step by step. Put your final answer in \\boxed{}."},
        {"role": "user", "content": "Find the number of ordered pairs (x, y) of positive integers with x + 2y = 2xy."},
    ],
    extra_body={"max_tokens": 8192, "tts_beam_size": 4, "tts_candidates_per_step": 4},
)
print(response.choices[0].message.content)
```

To route generation to a remote model rather than a local one, pass the key and endpoint
in the body: `extra_body={"tts_api_key": "sk-or-...", "model_base_url": "https://openrouter.ai/api/v1", ...}`.

### cURL

```bash
curl -X POST http://localhost:8001/v1/self_consistency/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek/deepseek-r1",
    "messages": [{"role": "user", "content": "What is 15 * 7?"}],
    "num_paths": 8,
    "max_tokens": 4096
  }'
```

## 5. Read the results

Strategy output beyond the standard message lives in a `tts_metadata` field on each
choice. It is not part of the OpenAI schema, so read it through `response.model_dump()`:

```python
meta = response.model_dump()["choices"][0]["tts_metadata"]
meta["selected_answer"]   # extracted final answer
meta["elapsed_time"]      # server-side wall-clock (seconds)
# self_consistency: consensus_score, answer_distribution
# beam_search / *_bon: steps[i]["score"], all_trajectories[i]["score"], token_stats["tflops"]
```

See [`api_guide.md`](api_guide.md) for every field.

## 6. Using a GPU: local generation or the PRM scorer

Two things need a GPU: running a **local generation model** (instead of a remote API), and
the **`prm` scorer** (a reward model that re-ranks trajectories). You can use either or
both. A common single-GPU setup is "PRM-only": generation stays remote on OpenRouter, and
the GPU only runs the PRM.

Set these before starting the service (bare metal):

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # avoids CUDA init errors

# PRM scorer on the GPU (generation stays remote -> do NOT set VLLM_MODEL_PATH)
export PRM_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
export PRM_DEVICE=cuda:0
export PRM_USE_VLLM=true
export PRM_GPU_MEMORY_UTILIZATION=0.9

python service_app/main.py
```

For local generation too, also set `VLLM_MODEL_PATH` (e.g. `Qwen/Qwen2.5-7B-Instruct`).

With the PRM running, call `offline_bon/prm` and pass your OpenRouter key in
`extra_body["tts_api_key"]` so generation routes remotely while the PRM scores locally:

```python
client = OpenAI(base_url="http://localhost:8001/v1/offline_bon/prm", api_key="unused")
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": "<a math problem>"},
    ],
    extra_body={
        "tts_api_key": "sk-or-...",      # authenticates to OpenRouter AND routes generation there
        "tts_num_trajectories": 8,
        "tts_score_aggregation": "min",
        "max_tokens": 8192,
    },
)
meta = response.model_dump()["choices"][0]["tts_metadata"]
print(meta["selected_answer"], meta["aggregated_score"])
```

> The service reads the key from `extra_body["tts_api_key"]`, not the client's `api_key`.
> Without `tts_api_key` (or `model_base_url`), an `offline_bon` / vLLM request tries to
> load a local generation model and fails.

### Docker with a GPU

The committed compose file is API-only. Add `docker-compose.gpu.yml` to attach the GPU
and the PRM environment, then start with both files:

```yaml
services:
  thinkbooster:
    environment:
      - PRM_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
      - PRM_DEVICE=cuda:0
      - PRM_USE_VLLM=true
      - PRM_GPU_MEMORY_UTILIZATION=0.9
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
      - CUDA_VISIBLE_DEVICES=0
      - HF_HOME=/app/.hf
    volumes:
      - ~/.cache/huggingface:/app/.hf      # download model weights once
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

> `Qwen2.5-Math-PRM-7B` is a **math** PRM — well-suited to math, exploratory on code. It
> scores per reasoning step, so quality also depends on how cleanly the output splits
> into steps.

## 7. Key parameters

Set via the URL (strategy, scorer) or `extra_body` (everything). Full list in
[`api_guide.md`](api_guide.md).

| Param | Default | Meaning |
|---|---|---|
| `num_paths` | 5 | reasoning paths for `self_consistency` |
| `tts_num_trajectories` | 8 | trajectories for `offline_bon` |
| `tts_beam_size` | 4 | beams for `beam_search` |
| `tts_candidates_per_step` | 4 | candidates per step (`online_bon`, `beam_search`) |
| `tts_score_aggregation` | `min` | step -> trajectory score: `min` / `mean` / `max` / `product` / `last` |
| `tts_api_key` | — | key for remote generation; authenticates and routes to the remote backend |
| `model_base_url` | — | remote OpenAI-compatible endpoint (defaults to OpenRouter) |
| `temperature` | 0.7 | keep > 0 so sampled paths differ |
| `max_tokens` | 4096 | raise to 8192+ for long reasoning |

Environment variables: `OPENROUTER_API_KEY` (remote generation), `VLLM_MODEL_PATH` (local
generation), `PRM_MODEL_PATH` (PRM scorer), `PORT` (default 8001).

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `API key not set for provider: openrouter` | Set `OPENROUTER_API_KEY`, or pass `tts_api_key` in `extra_body`. |
| A request tries to load a local model / asks for `VLLM_MODEL_PATH` | Add `tts_api_key` (or `model_base_url`) to route generation remotely. |
| Your `OpenAI(api_key=...)` looks ignored | It is — the service reads `extra_body["tts_api_key"]`, not the Authorization header. |
| `CUDA driver initialization failed` | `export VLLM_WORKER_MULTIPROC_METHOD=spawn` and restart. |
| CUDA out of memory | Lower `PRM_GPU_MEMORY_UTILIZATION` (e.g. 0.7) or use a larger GPU. |
| Docker build fails on missing `lm-polygraph/` | Clone it into the repo root first (§2). |
| `ModuleNotFoundError: latex2sympy2` | `pip install latex2sympy2`. |
