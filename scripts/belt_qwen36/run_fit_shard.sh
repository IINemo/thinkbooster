#!/bin/bash
# One fit-split shard of the Qwen3.6 BELT generation: ./run_fit_shard.sh <gpu> <shard> <count>
source /mnt/data/users/vlad.smirnov/belt_qwen36/env.sh
# BELT_v3: BELT_v2 (indented-blank-line block stops, cap on chains in flight)
# plus edit 9, one vLLM request per chain. Output goes to fit<count>/shard<N>,
# so cuts with different shard counts never share a directory.
BELT_DIR=${BELT_DIR:-$B/BELT_v3}
export U4R_NATIVE_ROOT=$BELT_DIR/vendor/uncertainty4reasoning-native-thinking
export RL_PROMPT_CONTRACT=qwen3.6-35b-a3b-fp8-native-thinking-v1
export CUDA_VISIBLE_DEVICES=$1
# Each chain is one request, replayed through BELT's generator for its block
# ends. BELT's own loop re-sent the whole context every block (at most 128
# tokens) and ran at about 175 tokens/s per RTX 5880. Set to 0 to go back to it.
export BELT_ONE_REQUEST_CHAINS=${BELT_ONE_REQUEST_CHAINS:-1}
# vLLM schedules the shard's chains itself; at most this many run at once, so
# their contexts (about 7k tokens each on average) fit the 301k-token KV cache.
MAX_SEQS=${BELT_MAX_NUM_SEQS:-32}
# Only used by the block-by-block loop (BELT_ONE_REQUEST_CHAINS=0, or a chain
# whose replay falls back to it).
export BELT_MAX_ACTIVE_CHAINS=${BELT_MAX_ACTIVE_CHAINS:-32}
export BELT_LOG_EVERY_CHAINS=32
SNAP=/mnt/data/users/vlad.smirnov/hf_cache/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/c653bba46dfa06e9658dcc0b0dc87fe4685f14a2
OUT=$B/full/fit$3/shard$2
cd $BELT_DIR/scripts/source_variants/olmo_code
# Own process group, so the watchdog below can take vLLM's engine process too.
PYTHONPATH="$PWD:$BELT_DIR" setsid $PY -m rl_controller.scripts.single_cot_probe \
  --manifest $B/pool/manifest.jsonl --split fit --split-salt qwen36-belt-2026-11-27-v1 \
  --model Qwen/Qwen3.6-35B-A3B-FP8 --revision c653bba46dfa06e9658dcc0b0dc87fe4685f14a2 \
  --model-path $SNAP --prompt-contract qwen3.6-35b-a3b-fp8-native-thinking-v1 \
  --chains 4 --max-chains-per-question 4 --budget 16384 --block-cap 128 \
  --query-stride 4 --max-forced-tokens 32 --max-model-len 20480 --gpu-mem-util 0.92 \
  --max-num-seqs $MAX_SEQS \
  --work-shard-index $2 --work-shard-count $3 --seed 0 --out-dir $OUT &
pid=$!
# vLLM's engine process can keep the interpreter alive after the shard is
# written: shards 0 and 1 of fit64 sat 40+ minutes on idle GPUs past their
# summary.json, and every shard since has hung the same way. summary.json is
# the last file written, so once it exists the process gets 1 minute to exit
# on its own before its group is killed.
killed=0
while kill -0 $pid 2>/dev/null; do
  if [ -s $OUT/summary.json ]; then
    for _ in $(seq 6); do kill -0 $pid 2>/dev/null || break; sleep 10; done
    if kill -0 $pid 2>/dev/null; then
      echo "summary.json written, process still alive after 1 min: killing it"
      kill -- -$pid; sleep 15; kill -9 -- -$pid 2>/dev/null; killed=1
    fi
    break
  fi
  sleep 30
done
wait $pid
rc=$?
[ $killed = 1 ] && [ -s $OUT/summary.json ] && rc=0
echo EXIT=$rc
