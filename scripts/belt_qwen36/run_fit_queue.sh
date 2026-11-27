#!/bin/bash
# ./run_fit_queue.sh <gpu>: run this GPU's fit shards (gpu, gpu+2, ... of 16) in turn.
# A shard with a chains.jsonl is finished and skipped, so rerunning resumes.
cd /mnt/data/users/vlad.smirnov/belt_qwen36/full
for s in $(seq $1 2 15); do
  if [ -s fit/shard$s/chains.jsonl ]; then echo "shard $s done, skipping"; continue; fi
  echo "shard $s start $(date -Is)"
  ./run_fit_shard.sh $1 $s 16 > logs/shard$s.log 2>&1
  echo "shard $s end $(date -Is) $(tail -1 logs/shard$s.log)"
done
echo QUEUE_DONE
