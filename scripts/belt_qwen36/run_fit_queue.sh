#!/bin/bash
# ./run_fit_queue.sh <gpu> <shard> [<shard> ...]
# Run the listed fit shards (of 16) on one GPU, in turn. A shard with a
# chains.jsonl is finished and skipped, so rerunning the same line resumes.
# With WAIT_PID set, first wait for that process (a shard already running on
# this GPU) to exit.
cd /mnt/data/users/vlad.smirnov/belt_qwen36/full
gpu=$1
shift
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
for s in "$@"; do
  if [ -s fit/shard$s/chains.jsonl ]; then echo "shard $s done, skipping"; continue; fi
  echo "shard $s start $(hostname) gpu$gpu $(date -Is)"
  ./run_fit_shard.sh $gpu $s 16 > logs/shard$s.log 2>&1
  echo "shard $s end $(date -Is) $(tail -1 logs/shard$s.log)"
done
echo QUEUE_DONE
