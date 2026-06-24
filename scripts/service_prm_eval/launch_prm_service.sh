#!/usr/bin/env bash
# Launch the ThinkBooster service in PRM-only mode (generation on OpenRouter,
# PRM re-ranking on the local GPU). See docs/service/running_with_prm.md.
#
# Usage:
#   ./scripts/service_prm_eval/launch_prm_service.sh
#
# Run this on the GPU box. GPU 0 is the free A6000 in the test setup; the busy
# GPU 1 is left alone. Weights download to the HF cache on first run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --- PRM scorer (local, on GPU) ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # GPU 0 = free A6000
export PRM_MODEL_PATH="${PRM_MODEL_PATH:-Qwen/Qwen2.5-Math-PRM-7B}"
export PRM_DEVICE="${PRM_DEVICE:-cuda:0}"
export PRM_USE_VLLM="${PRM_USE_VLLM:-true}"
export PRM_GPU_MEMORY_UTILIZATION="${PRM_GPU_MEMORY_UTILIZATION:-0.9}"

# Generation runs on OpenRouter -> do NOT set VLLM_MODEL_PATH (no local gen model).
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"  # avoids CUDA init errors
export PORT="${PORT:-8001}"

echo "Launching ThinkBooster (PRM-only) on port ${PORT}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  PRM_MODEL_PATH=${PRM_MODEL_PATH}"
echo "  PRM_USE_VLLM=${PRM_USE_VLLM}  PRM_GPU_MEMORY_UTILIZATION=${PRM_GPU_MEMORY_UTILIZATION}"

exec python service_app/main.py
