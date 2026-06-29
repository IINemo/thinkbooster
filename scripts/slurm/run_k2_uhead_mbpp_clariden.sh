#!/bin/bash
# =============================================================================
# Offline Best-of-N — K2-Think-V2 + UHead — MBPP+ — on CSCS Clariden (GH200)
# =============================================================================
# Runs the new-UHead offline-BoN MBPP+ experiment with the thinking budget the
# UHead was trained on (medium=<think_fast>, low=<think_faster>).
#
# Environment: the uenv+venv stack set up under $SCRATCH (NOT conda):
#   uenv image pytorch/v2.9.1:v2  +  venv at $SCRATCH/venvs/tb
#
# Submit (full run, 378 problems, partition=normal, 2x GH200):
#   cd $SCRATCH/thinkbooster && mkdir -p logs
#   BUDGET=medium sbatch scripts/slurm/run_k2_uhead_mbpp_clariden.sh
#   BUDGET=low    sbatch scripts/slurm/run_k2_uhead_mbpp_clariden.sh
#
# Smoke test (2 problems, debug partition, ~20 min):
#   SUBSET=2 sbatch --partition=debug --time=00:20:00 \
#       scripts/slurm/run_k2_uhead_mbpp_clariden.sh
#
# Sweep N (best-of-N) by overriding N:
#   BUDGET=medium N=8 sbatch scripts/slurm/run_k2_uhead_mbpp_clariden.sh
#
# HF token (K2-Think-V2 may be gated): export HF_TOKEN=... before sbatch, or put
# it in ~/.hf_token (chmod 600). Never commit the token.
# =============================================================================
#SBATCH --job-name=k2_uhead_mbpp
#SBATCH --account=a0142
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---- experiment parameters (override via env, e.g. `BUDGET=low N=8 sbatch ...`) ----
BUDGET="${BUDGET:-${1:-medium}}"      # medium (<think_fast>) | low (<think_faster>)
N="${N:-4}"                           # best-of-N: number of trajectories per problem
SUBSET="${SUBSET:-}"                  # e.g. 2 for a smoke test; empty = full 378 problems
SEED="${SEED:-42}"
RESUME="${RESUME:-0}"                 # 1 to resume an interrupted run

REPO="${REPO:-$SCRATCH/thinkbooster}"
UENV_IMG="${UENV_IMG:-pytorch/v2.9.1:v2}"
VENV="${VENV:-$SCRATCH/venvs/tb}"

case "$BUDGET" in
  medium|low) ;;
  *) echo "ERROR: BUDGET must be 'medium' or 'low' (got '$BUDGET')"; exit 2;;
esac
CONFIG="experiments/offline_best_of_n/mbpp_plus/offline_bon_vllm_k2_think_v2_mbpp_plus_uhead_${BUDGET}"

# ---- HF / caches: the shared project store already holds K2-Think-V2 (271 GB) ----
export HF_HOME="${HF_HOME:-/capstor/store/cscs/swissai/a0142/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_token" ]]; then
  HF_TOKEN="$(cat "$HOME/.hf_token")"
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# ---- wandb offline (compute nodes have no outbound network for wandb) ----
export WANDB_MODE="${WANDB_MODE:-offline}"
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Avoid CUDA fragmentation OOM: across chunks the native HS capture leaves
# reserved-but-unallocated memory that fragments until a large alloc fails.
# expandable_segments lets the allocator reuse it. (vLLM honours this too.)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$REPO/logs"

# ---- assemble Hydra overrides ----
OVERRIDES="strategy.num_trajectories=${N} system.seed=${SEED}"
[[ -n "$SUBSET" ]] && OVERRIDES="$OVERRIDES dataset.subset=${SUBSET}"
RESUME_FLAG=""
[[ "$RESUME" == "1" ]] && RESUME_FLAG="--resume"

echo "=================================================================="
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "budget=$BUDGET  N=$N  subset=${SUBSET:-full}  seed=$SEED  resume=$RESUME"
echo "config=$CONFIG"
echo "HF_HOME=$HF_HOME  HF_TOKEN=${HF_TOKEN:+set}"
echo "=================================================================="

# Run inside the uenv (provides CUDA torch 2.9.1 for GH200) + our venv.
# cd into the repo root so: (a) Hydra runtime.cwd resolves prompt_file, and
# (b) `python scripts/...` puts scripts/ on sys.path -> utils.hook_hs_extension importable.
uenv run "$UENV_IMG" --view=default -- bash -lc "
  set -euo pipefail
  source '$VENV/bin/activate'
  cd '$REPO'
  echo \"python: \$(which python)\"
  python scripts/run_tts_eval.py --config-path='$REPO/config' --config-name=$CONFIG $RESUME_FLAG $OVERRIDES
"

EXIT_CODE=$?
echo "[$(date)] finished with exit code $EXIT_CODE"
exit $EXIT_CODE
