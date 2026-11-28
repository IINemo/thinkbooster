#!/bin/bash
# ./run_fit_queue.sh <gpu> <shard> [<shard> ...]
# Run the listed fit shards on one GPU, in turn. SHARD_COUNT (default 64) is
# the number of shards the fit split is cut into; shard N of that cut lives in
# fit<count>/shard<N>, logs in logs/fit<count>/. single_cot_probe keeps a whole
# shard in memory and writes it only at the end, so smaller shards bound what a
# crash or kill loses (64 shards: about 270 chains, 1 to 1.4 h each).
# A shard counts as finished once its summary.json exists: single_cot_probe
# writes it last, after chains.jsonl, so a shard killed mid-write is redone
# rather than kept truncated. Rerunning the same line resumes.
# With WAIT_PID set, first wait for that process (a shard already running on
# this GPU) to exit.
cd /mnt/data/users/vlad.smirnov/belt_qwen36/full
count=${SHARD_COUNT:-64}
dir=fit$count
mkdir -p $dir logs/$dir
gpu=$1
shift
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
for s in "$@"; do
  if [ -s $dir/shard$s/summary.json ]; then echo "shard $s/$count done, skipping"; continue; fi
  # One shard per directory: refuse to start one that another queue is running.
  if pgrep -u "$USER" -f "work-shard-index $s --work-shard-count $count " > /dev/null; then
    echo "shard $s/$count is already running elsewhere on this host, skipping"; continue
  fi
  rm -rf $dir/shard$s
  echo "shard $s/$count start $(hostname) gpu$gpu $(date -Is)"
  ./run_fit_shard.sh $gpu $s $count > logs/$dir/shard$s.log 2>&1
  rc=$?
  echo "shard $s/$count end $(date -Is) exit $rc $(tail -1 logs/$dir/shard$s.log)"
done
echo QUEUE_DONE
