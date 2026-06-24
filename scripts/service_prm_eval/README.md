# Service PRM evaluation — CoT vs offline best-of-N

Implements the test plan in [`thinkbooster_service_prm_test_exps.md`](../../thinkbooster_service_prm_test_exps.md)
on top of [`docs/service/running_with_prm.md`](../../docs/service/running_with_prm.md).

**What it does**

1. Compares **plain CoT** vs **PRM offline best-of-N** for generation with
   `openai/gpt-oss-20b` (on OpenRouter).
2. Runs on **Gaokao2023en** (math, 385) and **HumanEval+** (code, 164).
3. **Times** each strategy (client latency, server-side elapsed, wall-clock).

| Strategy | Where it runs | Generation | Scoring |
|---|---|---|---|
| `cot` (baseline) | direct → OpenRouter | 1 sample | answer extracted from output |
| `prm` | → ThinkBooster service `/v1/offline_bon/prm` | N samples on OpenRouter | local PRM (Qwen2.5-Math-PRM-7B) re-ranks |

Correctness uses the same evaluators as the offline pipeline:
`EvaluatorExactMatch` (`math_equal`) for Gaokao, `EvaluatorHumanEvalPlus`
(EvalPlus full test suite) for HumanEval+.

---

## Prerequisites (run on the GPU box)

- 1× NVIDIA GPU ≥24 GB free (the test box has 2× RTX A6000; **GPU 0 is free**).
- `OPENROUTER_API_KEY` set in the environment or in `.env` at the repo root.
  The harness reads it from there and forwards it to the service via
  `tts_api_key`; it is never hard-coded.
- Deps installed (`./setup.sh`), plus the harness needs `openai`, `datasets`,
  `evalplus`, and optionally `python-dotenv`.

---

## Step 1 — launch the service (terminal A / tmux)

```bash
tmux new -t thinkbooster      # so it survives disconnect
./scripts/service_prm_eval/launch_prm_service.sh
# health check from another shell:
curl http://localhost:8001/health    # {"status":"healthy", ...}
```

The launcher pins the PRM to **GPU 0** (`CUDA_VISIBLE_DEVICES=0`), uses vLLM for
the PRM, and does **not** set `VLLM_MODEL_PATH` (generation stays on OpenRouter).
PRM weights download on first run (a few minutes). Detach with `Ctrl-b d`.

> Override any env inline, e.g. `PRM_GPU_MEMORY_UTILIZATION=0.7 ./scripts/service_prm_eval/launch_prm_service.sh`
> if you hit CUDA OOM (GPU 1 is busy in the test box, but GPU 0 is isolated by
> `CUDA_VISIBLE_DEVICES`).

---

## Step 2 — run the evaluation (terminal B)

Full plan (both datasets, both strategies, full subsets):

```bash
python scripts/service_prm_eval/run_service_prm_eval.py \
  --datasets gaokao humaneval \
  --strategies cot prm \
  --service-url http://localhost:8001 \
  --n 8 --agg min \
  --concurrency 8 \
  --resume
```

Smoke test first (recommended — validates the whole pipeline cheaply):

```bash
python scripts/service_prm_eval/run_service_prm_eval.py --limit 5 --concurrency 4
```

Just the baseline / just PRM:

```bash
# baseline only (no service needed)
python scripts/service_prm_eval/run_service_prm_eval.py --strategies cot
# PRM only
python scripts/service_prm_eval/run_service_prm_eval.py --strategies prm
```

Dataset-load sanity check (no API calls, no GPU):

```bash
python scripts/service_prm_eval/run_service_prm_eval.py --dry-run --limit 3
```

### Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--datasets` | `gaokao humaneval` | which datasets to run |
| `--strategies` | `cot prm` | `cot` (baseline) and/or `prm` (best-of-N) |
| `--n` | `8` | trajectories per problem for PRM best-of-N |
| `--agg` | `min` | PRM step→trajectory aggregation (`min/mean/max/product/last`) |
| `--subset-gaokao` / `--subset-humaneval` | `385` / `164` | per-dataset size |
| `--limit` | — | global cap per dataset (for quick runs) |
| `--concurrency` | `8` | parallel in-flight requests |
| `--temperature` / `--max-tokens` | `0.7` / `8192` | sampling |
| `--resume` | off | skip already-completed problems in `samples.jsonl` |
| `--out` | `outputs/service_prm_eval/<ts>` | output directory |

---

## Outputs

```
outputs/service_prm_eval/<timestamp>/
├── report.md                     # CoT-vs-PRM accuracy + timing tables
├── summary.json                  # machine-readable, all cells
├── gaokao__cot/    samples.jsonl + summary.json
├── gaokao__prm/    samples.jsonl + summary.json
├── humaneval__cot/ samples.jsonl + summary.json
└── humaneval__prm/ samples.jsonl + summary.json
```

Each `samples.jsonl` record has the gold answer, model output, extracted/selected
answer, PRM `aggregated_score`, `is_correct`, and timing (`client_latency_s`,
`server_elapsed_s`, tokens). `report.md` reports per-cell accuracy, wall-clock,
median latency, and a CoT-vs-PRM delta + a **PRM/CoT wall-clock ratio** (expected
to be roughly N× — that latency/accuracy trade-off is the point of the experiment).

---

## Notes

- **Cost/time:** PRM best-of-N issues **N generations per problem**. Full run =
  `(385 + 164) × (1 + N)` OpenRouter generations. Start with `--limit` to estimate.
- `Qwen2.5-Math-PRM-7B` is a **math** PRM. It is well-suited to Gaokao; on
  HumanEval+ treat PRM re-ranking as **exploratory** (the scorer was not trained
  on code), as flagged in `running_with_prm.md`.
- The run is **resumable**: re-run with `--resume` and the same `--out` to pick up
  after an interruption (errored problems are retried; successful ones are skipped).
