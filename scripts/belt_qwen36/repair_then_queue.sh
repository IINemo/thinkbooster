#!/bin/bash
# ./repair_then_queue.sh <gpu> <wait_pid> <v1_shard> <shard> [<shard> ...]
# Wait for the v1 shard running on this GPU (wait_pid), add the probes it
# missed at indented blank lines (repair_block_stops.py, into fit/shard<N>_v2),
# then run this GPU's remaining shards on BELT_v2. A failed repair is logged
# and does not hold up generation; it can be rerun on its own.
gpu=$1; wait_pid=$2; v1=$3; shift 3
source /mnt/data/users/vlad.smirnov/belt_qwen36/env.sh
F=$B/full
R=$B/BELT_v2
SNAP=/mnt/data/users/vlad.smirnov/hf_cache/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/c653bba46dfa06e9658dcc0b0dc87fe4685f14a2
while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done
if [ -s $F/fit/shard$v1/chains.jsonl ]; then
  echo "repair shard $v1 start $(date -Is)"
  (
    export RL_PROMPT_CONTRACT=qwen3.6-35b-a3b-fp8-native-thinking-v1
    export U4R_NATIVE_ROOT=$R/vendor/uncertainty4reasoning-native-thinking
    export CUDA_VISIBLE_DEVICES=$gpu
    cd $R/scripts/source_variants/olmo_code
    PYTHONPATH="$PWD:$R" $PY $B/patch/repair_block_stops.py --shard-dir $F/fit/shard$v1 \
      --out-dir $F/fit/shard${v1}_v2 --manifest $B/pool/manifest.jsonl --model-path $SNAP
  ) > $F/logs/repair_shard$v1.log 2>&1
  echo "repair shard $v1 end $(date -Is) exit $?"
else
  echo "shard $v1 has no chains.jsonl; repair skipped"
fi
cd $F
exec ./run_fit_queue.sh $gpu "$@"
