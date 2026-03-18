# Local Job Scheduling

Run experiments on local GPU machines using [Task Spooler](https://github.com/justanhduc/task-spooler) — a lightweight alternative to SLURM with GPU-aware scheduling.

> **Full reference**: See [`scripts/local/README.md`](../../scripts/local/README.md) for all options, examples, and GPU scheduling details.

## Quick Start

```bash
# Install task spooler
apt update && apt install task-spooler

# Submit an experiment
./scripts/local/submit.sh --strategy beam_search --dataset math500 --scorers entropy

# Submit with all scorers (auto-queues based on GPU availability)
./scripts/local/submit.sh --strategy offline_bon --dataset olympiadbench --scorers all

# Multiple seeds
./scripts/local/submit.sh --strategy baseline --dataset aime2025 --seeds 3

# Preview without submitting
./scripts/local/submit.sh --strategy online_bon --dataset gaokao2023en --scorers prm --dry-run
```

## Managing Jobs

```bash
tsp              # Show job queue
tsp -c <id>      # Show stdout of job
tsp -k <id>      # Kill running job
tsp -C           # Clear finished jobs
```

## Key Differences from SLURM

| | Local (tsp) | SLURM |
|---|---|---|
| Submit | `./scripts/local/submit.sh` | `./scripts/slurm/submit.sh` |
| GPU scheduling | tsp tracks GPU memory | SLURM `--gres=gpu:N` |
| Array jobs | Separate queued jobs | `--array=0-N` |
| Dependencies | `tsp -D <job_id>` | `--dependency=afterok:ID` |
