# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ThinkBooster is a framework for **test-time compute scaling** of LLM reasoning. It implements ~10 scaling strategies (beam search, best-of-N, self-consistency, DeepConf, MUR, phi-decoding, uncertainty CoT, adaptive scaling, extended thinking, baseline), scored by PRMs / uncertainty estimators / LLM-as-a-critic. It ships an evaluation pipeline (math/science/coding benchmarks), an OpenAI-compatible REST gateway, and an interactive visual debugger.

The PyPI package and import name is `thinkbooster` (recently renamed from `llm_tts`). The upstream repo is `IINemo/thinkbooster`.

## Mandatory review gate

**Always call the codex MCP connector to review before acting** — code changes, eval/run commands, config edits, shell scripts, anything load-bearing. Run it as an independent second pass *before* committing, opening a PR, or launching an experiment. It catches mistakes the immediate-context author misses, saves wasted compute on broken runs, and keeps decisions sound. Surface any blocker findings before proceeding; don't silently dismiss them — if you skip a finding, say which and why.

## Commands

```bash
# Install (core); add scorers that need GitHub-only deps (UHead, KernelAct, speculators)
pip install -e ".[dev]"        # dev: pytest, black, isort, flake8, pre-commit
./setup.sh                      # optional: llm-uncertainty-head, vllm-speculators, KernelAct
pip install -e ".[service]"     # FastAPI service deps

# Lint / format (targets: thinkbooster scripts service_app)
make lint                       # flake8 (max-line-length 100, ignores E203/W503/E501)
make format                     # black + isort (line-length 88, isort profile=black)
make fix                        # run all pre-commit hooks
make hooks                      # install pre-commit hooks

# Tests
make test                       # pytest tests/ -v
pytest tests/ -v                # integration tests EXCLUDED by default (-m "not integration")
pytest tests/deepconf/test_deepconf_accurate.py::test_name -v   # single test
pytest tests/service_app/test_integration.py -m integration -v  # integration (needs API keys)
python tests/strategy_registry.py --validate                    # CI gate — see below

# Run an experiment (Hydra; --config-name is a path under config/, no .yaml suffix)
python scripts/run_tts_eval.py \
  --config-name experiments/beam_search/gsm8k/window_all/mean/beam_search_vllm_qwen25_math_7b_instruct_gsm8k_prm \
  dataset.subset=3 report_to=''      # subset=N for quick smoke test, report_to='' to skip wandb
# add --resume to continue an interrupted run; results land in outputs/

# Service + visual debugger
python service_app/main.py          # http://localhost:8001 , debugger at /debugger
```

`black` uses line-length **88**; `flake8` allows **100** and ignores E501. Don't "fix" line length to satisfy flake8 — match black.

## Architecture

Data flows: **strategy** drives the search, asking a **generator** (backend) for step candidates and a **scorer** to rank them, until a trajectory completes; then **evaluation** grades correctness.

- **Strategies** (`thinkbooster/strategies/`) — all subclass `StrategyBase` and implement the single required method `generate_trajectories_batch(requests, sample_indices, save_callback)`. `generate_trajectory()` is a convenience wrapper around the batch method; override only for specialized single-sample behavior. Strategies check `self.cancel_event` between steps (used by the service to abort mid-generation). DeepConf lives in its own subpackage `strategies/deepconf/`.
- **Generators** (`thinkbooster/generators/`) — backend abstraction so strategies are backend-agnostic: `api.py` (OpenAI-compatible: OpenRouter/DeepSeek/etc.), `vllm.py` (local GPU), `huggingface.py`. They emit `StepCandidate` objects (`base.py`). White-box strategies (DeepConf, MUR, phi, uncertainty CoT) need logprobs/prefill, so they require the vLLM or HF backend, not arbitrary black-box APIs.
- **Scorers** (`thinkbooster/scorers/`) — `step_scorer_*.py` (PRM, uncertainty/entropy/perplexity, confidence, LLM-critic). `multi_scorer.py` composes several; aggregation (min/mean/max/product) and sliding window are configurable. Uncertainty scorers build on `lm-polygraph`.
- **Step boundary detectors** (`thinkbooster/step_boundary_detectors/`) — split generation into steps and find the final answer. `non_thinking/` uses structured markers (`- Step`, `<Answer>:`); `thinking/` handles `<think>…</think>` models. Strategy step-counting differs by mode (see `count_reasoning_steps` in `strategy_base.py`: thinking mode excludes the answer step).
- **Evaluation** (`thinkbooster/evaluation/`) — `exact_match.py` ports Qwen2.5-Math grading; `llm_as_a_judge.py` does multi-vote LLM verification; `human_eval_plus_evaluator.py`/`mbpp_plus_evaluator.py` run code via EvalPlus; `latex2sympy/` is vendored and excluded from lint/format/isort.
- **Service** (`service_app/`) — FastAPI gateway. Strategy + scorer are encoded in the **URL path** (e.g. `/v1/beam_search/prm/chat/completions`), so switching strategy means changing the URL, not the client code. `core/strategy_manager.py` is the strategy factory/lifecycle; `static/debugger/` is the web UI.

## Configuration (Hydra)

Experiment configs in `config/experiments/<strategy>/<dataset>/…` are entry points that **compose** modular configs via `defaults:` from `config/{dataset,model,strategy,scorer,evaluation,generation,prompts,system}/`. Override anything on the CLI (`dataset.subset=10`, `model.model_path=...`, `report_to=''`).

Experiment config filenames follow an **enforced** convention (tested by `tests/test_config_naming.py`):
```
{strategy_prefix}_{backend}_{model}_{dataset}_{scorer}.yaml
```
e.g. `offline_bon_openrouter_gpt4o_mini_math500_entropy.yaml`. Dataset must appear in full (no abbreviations) and come before the scorer. If you add a new model/dataset/scorer, update the `KNOWN_*` sets in that test or CI fails.

## Adding a strategy (test gate)

CI runs `python tests/strategy_registry.py --validate` **before** pytest and blocks merge if a registered strategy is missing its required test files. To add one: implement under `thinkbooster/strategies/`, create `tests/<name>/` with the required test files (each containing at least one `def test_*`), then add a `StrategyInfo` entry to `REGISTERED_STRATEGIES` in `tests/strategy_registry.py`. Use `--template <name>` to scaffold the registry entry, `--list` to inspect. Currently only `deepconf` is registered with dedicated tests; the rest are covered by the eval pipeline + integration tests. See `docs/core/strategy_registration.md`.

## Conventions & gotchas

- Integration tests are marked `@pytest.mark.integration` and skipped by default; they read `OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` (copy `.env.example` → `.env`). pytest runs with `--strict-markers` and coverage on `thinkbooster` + `service_app`.
- `scripts/run_tts_eval.py` sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` and the multiprocessing start method **before any CUDA import** — keep new imports below that block (the `# flake8: noqa: E402` at the top is intentional).
- Checkpoints/results: `outputs/` dirs are per-host and transient. Important checkpoints live on Hugging Face (`karantonis/bayesclue-checkpoints`) — check HF first, push anything worth keeping.
- This repo is shared between local and GPU servers via the same git: edit locally → push → pull on the server. Never run from uncommitted changes; confirm the server branch first.
