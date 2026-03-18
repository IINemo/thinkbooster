# Scripts

## Core

| Script | Description |
|--------|-------------|
| `run_tts_eval.py` | Main evaluation pipeline — dataset loading, strategy execution, scoring, metrics |
| `generate_cached_examples.py` | Generate `cached_examples.json` for the visual debugger |
| `utils/results.py` | Shared result loading utilities |

## Analysis (`analysis/`)

Post-hoc analysis and re-evaluation of experiment results.

| Script | Description |
|--------|-------------|
| `reeval.py` | Re-run the full evaluation phase from a checkpoint (`batch_results.jsonl` + Hydra config) |
| `evaluate_results.py` | Run specific evaluators on existing results with custom config; includes correlation analysis |
| `run_llm_judge_local.py` | Standalone LLM judge via OpenAI API — no Hydra needed, supports majority voting |
| `aggregate_seed_results.py` | Aggregate statistics across multi-seed experiment runs |
| `analyze_candidates.py` | Post-hoc analysis of multi-scorer offline best-of-N candidates |
| `analyze_garbage_generation.py` | Detect degenerate token output from wandb run logs |
| `analyze_thinking_steps.py` | Compare step boundary detectors on thinking model trajectories |
| `probe_model_capabilities.py` | Test model support for logprobs and prefill across providers |

## Data (`data/`)

Dataset conversion, upload, and logging.

| Script | Description |
|--------|-------------|
| `convert_datasets.py` | Convert datasets to unified format for evaluation |
| `upload_to_hf.py` | Upload converted datasets to HuggingFace Hub |
| `log_results_to_wandb.py` | Upload existing experiment results to Weights & Biases |

## Job Submission

| Directory | Description |
|-----------|-------------|
| `local/` | Local GPU job scheduling via [Task Spooler](https://github.com/justanhduc/task-spooler) — see [local/README.md](local/README.md) |
| `slurm/` | SLURM cluster job submission — see [slurm/README.md](slurm/README.md) |
