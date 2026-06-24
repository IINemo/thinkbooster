# Running the ThinkBooster Service with the PRM Scorer

How to launch the ThinkBooster service so you can hit it over an OpenAI-compatible
API and run **offline best-of-N with a PRM scorer**, where:

- **generation** happens **remotely** on OpenRouter (e.g. `openai/gpt-oss-20b`), and
- the **PRM reward model** runs **locally on one GPU** and re-ranks the sampled trajectories.

This is the recommended single-GPU setup ("PRM-only mode"). You do **not** need a local
generation model — no `VLLM_MODEL_PATH`. The only thing the GPU does is run the PRM.

```
                       extra_body={"tts_api_key": "<your OpenRouter key>"}
  your client ──HTTP──►  ThinkBooster service  ──── generates N trajectories ────►  OpenRouter
 (openai SDK / curl)     /v1/offline_bon/prm                                       (gpt-oss-20b)
                                │
                                │  scores each trajectory with a local PRM (GPU)
                                ▼
                         Qwen2.5-Math-PRM-7B  ── picks the best trajectory ──► response
```

---

## 1. Prerequisites

| Need | Detail |
|---|---|
| GPU | 1× NVIDIA GPU, **≥24 GB** (PRM-7B in bf16 ≈ 15 GB + KV cache). 3090/4090/A6000/A100/H100 all work. |
| NVIDIA driver | Required. For the Docker path you also need `nvidia-container-toolkit` installed on the host. |
| OpenRouter key | Generation runs on OpenRouter — get a key at https://openrouter.ai/keys |
| Tooling | `git` + **either** Docker **or** a Python 3.10+ environment. |

---

## 2. Get the code

```bash
git clone https://github.com/IINemo/thinkbooster.git
cd thinkbooster

# lm-polygraph is a separate repo (gitignored). It is required both for the
# Docker build and for the scorers. Clone it INTO the repo root:
git clone https://github.com/IINemo/lm-polygraph.git

# .env is copied for service config. The OpenRouter key here is OPTIONAL because each
# request carries its own key via tts_api_key (see §4). Set it only as a fallback.
cp service_app/.env.example .env
#   (optional) edit .env and set:  OPENROUTER_API_KEY=sk-or-...   # used only if a request omits tts_api_key
```

---

## 3. Launch the service — pick ONE option

> On a single-GPU box, **Option B (bare metal) is the quickest and is our deployment path.**
> Use **Option A (Docker)** if you want an isolated/reproducible container.

### Option A — Docker (GPU-enabled)

The committed `docker-compose.yml` is API-only (no GPU). Add a small override file that
attaches the GPU and the PRM env. Create **`docker-compose.gpu.yml`** in the repo root:

```yaml
services:
  thinkbooster:
    environment:
      # --- PRM scorer (local, on GPU) ---
      - PRM_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
      - PRM_DEVICE=cuda:0
      - PRM_USE_VLLM=true
      - PRM_GPU_MEMORY_UTILIZATION=0.9
      # generation is on OpenRouter, so DO NOT set VLLM_MODEL_PATH
      - VLLM_WORKER_MULTIPROC_METHOD=spawn   # avoids CUDA init errors
      - CUDA_VISIBLE_DEVICES=0
      - HF_HOME=/app/.hf                     # cache model weights on the mounted volume
    volumes:
      - ~/.cache/huggingface:/app/.hf        # reuse HF cache → PRM weights downloaded once
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Build and start (PRM weights download on first run — a few minutes):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d

# health + logs
curl http://localhost:8001/health          # {"status":"healthy", ...}
docker compose logs -f thinkbooster
```

Service is now on **`http://<server-ip>:8001`** (docs at `/docs`).

> **Build gotchas**
> - The build fails if `lm-polygraph/` is missing — clone it first (step 2).
> - If the build errors on `README.md` not found, add `COPY README.md .` to
>   `service_app/Dockerfile` just before the `pip install` line, then rebuild.

### Option B — Bare metal in tmux (single GPU, simplest)

```bash
# one-time: install deps + clone lm-polygraph + scorers
./setup.sh

# keep the service alive after you disconnect
apt update && apt install -y tmux
tmux new -t thinkbooster

# inside tmux — PRM-only mode (generation on OpenRouter, PRM on GPU)
export CUDA_VISIBLE_DEVICES=0
export PRM_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
export PRM_DEVICE=cuda:0
export PRM_USE_VLLM=true
export PRM_GPU_MEMORY_UTILIZATION=0.9
export VLLM_WORKER_MULTIPROC_METHOD=spawn      # avoids CUDA init errors
export PORT=8001                               # default bare-metal port is 8080
python service_app/main.py
```

Detach from tmux with `Ctrl-b d`. Logs are written to `logs/<date>/<time>/service.log`.

---

## 4. Send requests — offline best-of-N + PRM, generation on OpenRouter

> **⚠️ The one thing you must get right:** put your OpenRouter key in
> **`extra_body["tts_api_key"]`**, *not* in the `OpenAI(...)` client. The service ignores the
> client's `api_key` (the HTTP Authorization header) and reads `tts_api_key` from the body.
> That field does double duty: it authenticates you to OpenRouter **and** routes `offline_bon`
> to the OpenRouter backend (generation runs remotely instead of on a local model). Passing
> only `provider=openrouter` is **not** enough — without `tts_api_key` (or `model_base_url`)
> the service tries to load a local generation model and fails.

Strategy + scorer are set via the **URL path** (`/v1/offline_bon/prm`); everything else
goes in `extra_body`.

### Python (math example — Gaokao)

```python
import time
from openai import OpenAI

OPENROUTER_KEY = "sk-or-..."   # your OpenRouter key

client = OpenAI(
    base_url="http://<server-ip>:8001/v1/offline_bon/prm",  # strategy + scorer in the path
    api_key="unused",   # IGNORED by the service — the real key goes in extra_body below
)

t0 = time.perf_counter()
resp = client.chat.completions.create(
    model="openai/gpt-oss-20b",                # OpenRouter model slug
    messages=[
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": "<a Gaokao problem here>"},
    ],
    extra_body={
        "tts_api_key": OPENROUTER_KEY,    # ← your key: authenticates AND routes to OpenRouter
        "tts_num_trajectories": 8,        # N — number of sampled trajectories
        "tts_score_aggregation": "min",   # how per-step PRM scores collapse to one number
        "temperature": 0.7,
        "max_tokens": 8192,
    },
)
elapsed = time.perf_counter() - t0

meta = resp.model_dump()["choices"][0]["tts_metadata"]
print(resp.choices[0].message.content)     # best trajectory (full reasoning)
print("answer:", meta["selected_answer"])
print("winning score:", meta.get("aggregated_score"))
print("server elapsed_time:", meta["elapsed_time"], "s")   # server-side wall-clock
print("client elapsed:", round(elapsed, 2), "s")
# every candidate + its PRM score:
for i, tr in enumerate(meta.get("all_trajectories", [])):
    print(f"  traj {i}: score={tr['score']:.3f}")
```

### Python (code example — HumanEval+)

Same call, only the model is code-oriented so use the HumanEval system prompt:

```python
messages=[
    {"role": "system", "content":
        "Do not output any other code except for asked self-contained Python script. "
        "Do not provide any guides on how to run it. Code will be parsed from codeblock "
        "to check it's correct."},
    {"role": "user", "content": "<a HumanEval+ prompt here>"},
]
```

> Note: `Qwen2.5-Math-PRM-7B` is a **math** PRM. On Gaokao it is well-suited; on HumanEval+
> treat PRM re-ranking as exploratory. PRM scores per **reasoning step**, so quality also
> depends on how cleanly the model's output splits into steps.

### cURL equivalent

```bash
curl -X POST http://<server-ip>:8001/v1/offline_bon/prm/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [
      {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
      {"role": "user", "content": "What is 15 * 7?"}
    ],
    "tts_api_key": "sk-or-...",
    "tts_num_trajectories": 8,
    "tts_score_aggregation": "min",
    "max_tokens": 8192
  }'
```

---

## 5. Baseline for the comparison (plain CoT, no TTS)

For "обычный CoT vs PRM offline best-N", run a single direct generation against the same
model — bypass the service or call `self_consistency` with `num_paths: 1`. Cleanest is a
direct OpenRouter call:

```python
import time
from openai import OpenAI

base = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="sk-or-...")  # your OpenRouter key, direct

t0 = time.perf_counter()
resp = base.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": "<same problem>"},
    ],
    temperature=0.7, max_tokens=8192,
)
print("CoT elapsed:", round(time.perf_counter() - t0, 2), "s")
print(resp.choices[0].message.content)
```

Best-of-N does **N** generations, so expect its latency to be roughly **N×** the CoT
baseline — that latency/accuracy trade-off is exactly what we're measuring.

---

## 6. Parameters that matter

| Param (in `extra_body`) | Default | Meaning |
|---|---|---|
| `model` (top-level) | — | OpenRouter slug, e.g. `openai/gpt-oss-20b` |
| `tts_api_key` | — | **Required** — your OpenRouter key; authenticates *and* routes generation to OpenRouter |
| `model_base_url` | optional | Point at a non-OpenRouter OpenAI-compatible endpoint; defaults to OpenRouter when `tts_api_key` is set |
| `tts_num_trajectories` | 8 | N — trajectories sampled, then PRM-ranked |
| `tts_score_aggregation` | `min` | per-step → trajectory score: `min`/`mean`/`max`/`product`/`last` |
| `temperature` | 0.7 | sampling temperature (keep > 0 so the N trajectories differ) |
| `max_tokens` | 4096 | raise to 8192+ for long reasoning |

Read results from `resp.model_dump()["choices"][0]["tts_metadata"]`:
`selected_answer`, `aggregated_score`, `all_trajectories` (each with `score`),
and `elapsed_time` (server-side seconds). See `docs/service/api_guide.md` for the full
API reference.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `API key not set for provider: openrouter` | Missing `tts_api_key` in `extra_body` (and no server-side `OPENROUTER_API_KEY` fallback). |
| Request loads a local model / asks for `VLLM_MODEL_PATH` | You forgot `tts_api_key` (or `model_base_url`) in `extra_body` — add it (see §4). |
| Your `OpenAI(api_key=...)` looks ignored | It is — the service reads the key from `extra_body["tts_api_key"]`, not the Authorization header. |
| `CUDA driver initialization failed` | `export VLLM_WORKER_MULTIPROC_METHOD=spawn` and restart. |
| CUDA OOM | Lower `PRM_GPU_MEMORY_UTILIZATION` (e.g. 0.7) or use a bigger GPU. |
| `ModuleNotFoundError: latex2sympy2` | `pip install latex2sympy2` (bare metal). |
| `operator torchvision::nms does not exist` | `pip install torchvision --upgrade`. |
| Docker build fails at `COPY lm-polygraph/` | Clone `lm-polygraph` into the repo root first (§2). |
