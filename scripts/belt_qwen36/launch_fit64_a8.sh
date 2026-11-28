#!/bin/bash
# Re-split the remaining 64-way shards over three GPUs: Artem-8 GPU0 takes the
# even shards 12-46, GPU1 the odd shards 13-45, Artem-1 GPU1 takes 47-63.
# Each new queue first waits for the shard already running on its GPU.
cd /mnt/data/users/vlad.smirnov/belt_qwen36/full
w0=$(pgrep -u "$USER" -f "^/bin/bash ./run_fit_shard.sh 0 10 64")
w1=$(pgrep -u "$USER" -f "^/bin/bash ./run_fit_shard.sh 1 11 64")
echo "waiting on [$w0] [$w1] $(date -Is)" > logs/queue64v4_wait.txt
WAIT_PID=$w0 SHARD_COUNT=64 setsid nohup ./run_fit_queue.sh 0 $(seq -s " " 12 2 46) > logs/queue64v4_g0.log 2>&1 < /dev/null &
WAIT_PID=$w1 SHARD_COUNT=64 setsid nohup ./run_fit_queue.sh 1 $(seq -s " " 13 2 45) > logs/queue64v4_g1.log 2>&1 < /dev/null &
