# K2-Think-V2 + UHead — Offline Best-of-N on MBPP+ (CSCS Clariden)

Offline best-of-N test-time scaling with **K2-Think-V2** generating and the **new
UHead** checkpoint scoring, on **MBPP+**, run on the CSCS Clariden GH200 cluster.

## What this runs

For each MBPP+ problem: generate `N` complete thinking trajectories, split each
post-hoc into reasoning steps, score every step with the UHead uncertainty head,
and select the trajectory with the best (minimum) step score. Correctness is
graded by the EvalPlus MBPP+ suite.

- **Generator:** `LLM360/K2-Think-V2` (vLLM, tensor_parallel_size=2, native thinking mode)
- **Scorer (UHead):** `rediska0123/uhead_hs_K2-Think-V2_mixed_code10K_steps_vllm_10epochs`
  (the new checkpoint), via native hidden-state capture (`utils.hook_hs_extension`).
- **Dataset:** `evalplus/mbppplus`, `test` split (378 problems).

## The thinking-budget prompts (medium / low)

The UHead author trained on two K2-Think completion prefixes and recommends
running with them. They differ **only** in the thinking-budget token:

| Variant | Budget token | Reproduced by |
|---------|--------------|---------------|
| medium  | `<think_fast>`   | `model.reasoning_effort: medium` |
| low     | `<think_faster>` | `model.reasoning_effort: low`    |

This is **not** hand-injected text — K2-Think-V2's chat template maps
`reasoning_effort` directly to those tokens (verified against the tokenizer:
`medium → <think_fast>`, `low → <think_faster>`, `high → <think>`). Combined with
the user-message wrapper in [`config/prompts/k2_think_answer.txt`](../../config/prompts/k2_think_answer.txt)
(`"Answer the following question. Enclose your answer in <answer></answer>. <Question>: {question}"`),
the chat template reproduces the author's training prefixes **exactly**, with no
double-wrapping. The author's original raw prefix files
(`k2_think_{medium,low}_completion_prefix.txt`) are the reference these were
matched against.

## Configs

- `config/experiments/offline_best_of_n/mbpp_plus/offline_bon_vllm_k2_think_v2_mbpp_plus_uhead_medium.yaml`
- `config/experiments/offline_best_of_n/mbpp_plus/offline_bon_vllm_k2_think_v2_mbpp_plus_uhead_low.yaml`

Both are copies of the PR #253 `..._uhead.yaml` config with three changes: the new
`scorer.uq_head_path`, `model.reasoning_effort`, and `dataset.prompt_file`.

## Prerequisites (one-time, on the cluster)

1. The base env: uenv `pytorch/v2.9.1:v2` + venv `$SCRATCH/venvs/tb`
   (torch 2.9.1 CUDA for GH200, vLLM 0.12, thinkbooster installed).
2. The UHead runtime stack in that venv (GitHub-only deps):
   `lm-polygraph` (dev), `vllm-speculators`, `llm-uncertainty-head` (`luh`).
3. `HF_HOME=/capstor/store/cscs/swissai/a0142/hf_cache` (shared project store;
   already holds K2-Think-V2). Export `HF_TOKEN` or put it in `~/.hf_token`.

## Run

```bash
cd $SCRATCH/thinkbooster && mkdir -p logs

# Smoke test first (2 problems, debug partition, ~20 min):
SUBSET=2 sbatch --partition=debug --time=00:20:00 scripts/slurm/run_k2_uhead_mbpp_clariden.sh

# Full runs (378 problems, partition=normal, 12 h, 2x GH200):
BUDGET=medium sbatch scripts/slurm/run_k2_uhead_mbpp_clariden.sh
BUDGET=low    sbatch scripts/slurm/run_k2_uhead_mbpp_clariden.sh
```

Override knobs via env: `BUDGET` (medium|low), `N` (best-of-N, default 4),
`SUBSET`, `SEED`, `RESUME=1`. Results land in `outputs/` with full config
snapshots; wandb runs offline.
