#!/usr/bin/env bash
# Run BELT tutorial jobs one after another on one GPU.
#
#   bash scripts/belt_tutorial/run_queue.sh <gpu> <queue-file>
#
# Each queue line is "<dataset> <policy-tag> <seed> [extra hydra overrides...]".
# <dataset> names config/experiments/belt/tutorial/<dataset>_qwen3_1p7b.yaml and
# <policy-tag> names $BELT_TUTORIAL_ROOT/policies/<policy-tag>.json. Generation
# and the BELT reader share the GPU. A finished run (results.json present) is
# skipped, so a queue can be restarted after a failure.
set -uo pipefail

GPU=$1
QUEUE=$2
ROOT=${BELT_TUTORIAL_ROOT:-/mnt/data/users/vlad.smirnov/belt_tutorial}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${BELT_ENV:-/mnt/data/users/vlad.smirnov/conda_envs/thinkbooster-belt}"
export HF_HOME=${HF_HOME:-/mnt/data/users/vlad.smirnov/hf_cache}
export TOKENIZERS_PARALLELISM=false
cd "$REPO"

while read -r dataset tag seed extra; do
    [[ -z "${dataset:-}" || "$dataset" == \#* ]] && continue
    out="$ROOT/runs/$dataset/$tag/seed$seed"
    if ls "$out"/*/results.json >/dev/null 2>&1 || [ -f "$out/results.json" ]; then
        echo "=== $(date +%F_%T) skip $dataset $tag seed$seed (done)"
        continue
    fi
    mkdir -p "$out"
    echo "=== $(date +%F_%T) start $dataset $tag seed$seed on GPU $GPU"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$GPU python scripts/run_tts_eval.py \
        --config-path "$REPO/config" \
        --config-name "experiments/belt/tutorial/${dataset}_qwen3_1p7b" \
        strategy.policy_path="$ROOT/policies/$tag.json" \
        system.seed="$seed" \
        scorer.device=cuda:0 \
        model.gpu_memory_utilization=0.55 \
        hydra.run.dir="$out" \
        $extra > "$out/stdout.log" 2>&1
    status=$?
    echo "=== $(date +%F_%T) end $dataset $tag seed$seed status=$status"
done < "$QUEUE"
echo "=== $(date +%F_%T) queue $QUEUE finished"
