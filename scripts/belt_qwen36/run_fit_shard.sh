#!/bin/bash
# One fit-split shard of the Qwen3.6 BELT generation: ./run_fit_shard.sh <gpu> <shard> <count>
source /mnt/data/users/vlad.smirnov/belt_qwen36/env.sh
# BELT_v2: the patch with the indented-blank-line block stops (shards from
# 2026-11-27 22:10 on). Shards 0, 1, 6, 7 ran on BELT (v1) and are repaired.
BELT_DIR=${BELT_DIR:-$B/BELT_v2}
export U4R_NATIVE_ROOT=$BELT_DIR/vendor/uncertainty4reasoning-native-thinking
export RL_PROMPT_CONTRACT=qwen3.6-35b-a3b-fp8-native-thinking-v1
export CUDA_VISIBLE_DEVICES=$1
# At most 32 chains in flight (about 216k tokens of context in steady state,
# against a 301k-token KV cache), so each round's prefix stays cached. The
# first four shards ran all ~1,074 at once and re-read every context per round.
export BELT_MAX_ACTIVE_CHAINS=${BELT_MAX_ACTIVE_CHAINS:-32}
export BELT_LOG_EVERY_CHAINS=32
SNAP=/mnt/data/users/vlad.smirnov/hf_cache/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/c653bba46dfa06e9658dcc0b0dc87fe4685f14a2
cd $BELT_DIR/scripts/source_variants/olmo_code
PYTHONPATH="$PWD:$BELT_DIR" $PY -m rl_controller.scripts.single_cot_probe \
  --manifest $B/pool/manifest.jsonl --split fit --split-salt qwen36-belt-2026-11-27-v1 \
  --model Qwen/Qwen3.6-35B-A3B-FP8 --revision c653bba46dfa06e9658dcc0b0dc87fe4685f14a2 \
  --model-path $SNAP --prompt-contract qwen3.6-35b-a3b-fp8-native-thinking-v1 \
  --chains 4 --max-chains-per-question 4 --budget 16384 --block-cap 128 \
  --query-stride 4 --max-forced-tokens 32 --max-model-len 20480 --gpu-mem-util 0.92 \
  --work-shard-index $2 --work-shard-count $3 --seed 0 --out-dir $B/full/fit/shard$2
echo EXIT=$?
