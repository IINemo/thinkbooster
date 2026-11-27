#!/bin/bash
# ./setup_and_queue.sh <gpu> <shard> [<shard> ...]
# On a host that already has the venv, BELT checkout and pool: download the
# pinned model from Hugging Face (resumes; a lock keeps a host's two GPUs from
# downloading at once), check that no file is a dangling link, then run the
# listed fit shards on this GPU.
V=/mnt/data/users/vlad.smirnov
REV=c653bba46dfa06e9658dcc0b0dc87fe4685f14a2
SNAP=$V/hf_cache/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/$REV
set -e
mkdir -p $V/hf_cache
HF_HOME=$V/hf_cache HF_HUB_DISABLE_TELEMETRY=1 flock $V/hf_cache/.qwen36-download.lock \
  $V/venvs/belt-qwen36/bin/hf download Qwen/Qwen3.6-35B-A3B-FP8 --revision $REV > /dev/null
dangling=$(find $SNAP -xtype l | wc -l)
if [ "$dangling" != 0 ]; then echo "model has $dangling dangling links"; exit 1; fi
echo "model $(du -shL $SNAP | cut -f1) ready on $(hostname) $(date -Is)"
cd $V/belt_qwen36/full
exec ./run_fit_queue.sh "$@"
