#!/bin/bash
# ./run_fit_queue.sh <gpu> <shard> [<shard> ...]
# Run the listed fit shards (of 16) on one GPU, in turn. A shard counts as
# finished once its summary.json exists: single_cot_probe writes it last, after
# chains.jsonl, so a shard killed mid-write is redone rather than kept
# truncated. Rerunning the same line resumes.
# With WAIT_PID set, first wait for that process (a shard already running on
# this GPU) to exit.
cd /mnt/data/users/vlad.smirnov/belt_qwen36/full
gpu=$1
shift
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
for s in "$@"; do
  if [ -s fit/shard$s/summary.json ]; then echo "shard $s done, skipping"; continue; fi
  # One shard per directory: refuse to start one that another queue is running.
  if pgrep -u "$USER" -f "work-shard-index $s --work-shard-count 16" > /dev/null; then
    echo "shard $s is already running elsewhere on this host, skipping"; continue
  fi
  rm -rf fit/shard$s
  echo "shard $s start $(hostname) gpu$gpu $(date -Is)"
  ./run_fit_shard.sh $gpu $s 16 > logs/shard$s.log 2>&1
  rc=$?
  echo "shard $s end $(date -Is) exit $rc $(tail -1 logs/shard$s.log)"
done
echo QUEUE_DONE
