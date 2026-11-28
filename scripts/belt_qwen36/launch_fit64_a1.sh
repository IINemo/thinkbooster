#!/bin/bash
# Artem-1 GPU1 takes 64-way fit shards 47-63 (Artem-8 runs 0-46). Waits for the
# model download, then runs with the same vLLM sampler path as Artem-8
# (VLLM_USE_FLASHINFER_SAMPLER=0), so every one-request shard draws the same way.
cd /mnt/data/users/vlad.smirnov/belt_qwen36/full
d=$(pgrep -u "$USER" -f "hf download Qwen/Qwen3.6-35B-A3B-FP8")
echo "waiting on download [$d] $(date -Is)" > logs/queue64a1_wait.txt
while [ -n "$d" ] && kill -0 $d 2>/dev/null; do sleep 30; done
S=/mnt/data/users/vlad.smirnov/hf_cache/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/c653bba46dfa06e9658dcc0b0dc87fe4685f14a2
n=$(ls $S/*.safetensors 2>/dev/null | wc -l)
echo "download done $(date -Is): $n safetensors files" >> logs/queue64a1_wait.txt
[ "$n" -ge 42 ] || { echo "model incomplete, not starting" >> logs/queue64a1_wait.txt; exit 1; }
VLLM_USE_FLASHINFER_SAMPLER=0 SHARD_COUNT=64 exec ./run_fit_queue.sh 1 $(seq -s " " 47 63)
