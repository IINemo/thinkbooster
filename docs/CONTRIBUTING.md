# Contributing to ThinkBooster

Thank you for your interest in contributing! This guide covers environment setup, development workflow, and how to add new components.

---

## Step 1: Understand the Project

Start by reading the core documentation:

- **[Project Structure](PROJECT_STRUCTURE.md)** — architecture overview, components, and design patterns
- **[Strategy Registration](STRATEGY_REGISTRATION.md)** — how to add new strategies with tests
- **[DeepConf Guide](deepconf/DeepConf.md)** — example strategy implementation

Quick architecture overview:

```
llm_tts/strategies/     → TTS strategy implementations
llm_tts/models/         → Model wrappers with streaming support
llm_tts/scorers/        → Step scoring functions (PRM, uncertainty)
llm_tts/evaluation/     → Correctness evaluation methods
config/                 → Hydra configuration system
tests/                  → Test suite with strategy registry
```

---

## Step 2: Set Up Development Environment

```bash
# Clone the repository
git clone https://github.com/IINemo/thinkbooster.git
cd thinkbooster

# Create and activate conda environment
conda create -n thinkbooster python=3.11 -y
conda activate thinkbooster

# Install all dependencies (lm-polygraph, vLLM, latex2sympy2, etc.)
./setup.sh

# Install dev dependencies and git hooks
pip install -e ".[dev]"
make hooks
```

**What this does:**

- Creates isolated conda environment with Python 3.11
- `setup.sh` installs the package in editable mode with vLLM, lm-polygraph, latex2sympy2, and all dependency pins
- `.[dev]` adds pytest, black, isort, flake8
- Sets up pre-commit hooks (black, isort, flake8)

### Configure HuggingFace Cache (Recommended)

If your machine has limited home directory space, point the HuggingFace cache to a larger disk:

```bash
echo 'export HF_HOME=/path/to/large/disk/.cache/huggingface' >> ~/.bashrc
source ~/.bashrc
```

The first experiment run will download models (~16GB each for 7B models). Subsequent runs use the cached versions.

---

## Step 3: Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your keys:
# OPENROUTER_API_KEY=sk-or-v1-...
# DEEPSEEK_API_KEY=sk-...
```

**Required for:**
- Running experiments with API models (OPENROUTER_API_KEY)
- Evaluation with LLM judge (DEEPSEEK_API_KEY or OPENROUTER_API_KEY)

---

## Step 4: Verify Installation

```bash
# Validate strategy registry
python tests/strategy_registry.py --validate

# Run all tests
pytest tests/ -v
```

Expected result: all tests pass (some may skip if API keys are not set).

---

## Step 5: Run Your First Experiment

```bash
# Quick test with 1 sample via API (no GPU needed)
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/offline_best_of_n/math500/offline_bon_openrouter_gpt4o_mini_math500_entropy \
  dataset.subset=1 \
  report_to=''

# Quick test with vLLM on GPU (requires 2 GPUs for generation + PRM scorer)
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/beam_search/math500/offline_bon_vllm_qwen25_math_7b_instruct_math500_entropy \
  dataset.subset=3 \
  report_to=''
```

### Output Structure

Results are saved to `outputs/<date>/<dataset>/<model>/<strategy>/seed<N>_<timestamp>/`:

- `results.json` — predictions and per-sample metrics
- `metrics.json` — summary metrics (accuracy, token usage)
- `run_tts_eval.log` — structured experiment log
- `stderr.log` — error log
- `.hydra/` — full config snapshot for reproducibility

### WandB Logging

Experiment configs have `report_to: wandb` enabled by default. To set up:

```bash
wandb login
# Optional: log under a shared organization
export WANDB_ENTITY=your-org-name
```

To disable wandb for local testing, add `report_to=''` to the command (as shown above).

### Validating a Running Experiment

After launching an experiment, check the log for these lines:

```
[INFO] - API key validated successfully: OPENROUTER_API_KEY
[INFO] - vLLM model loaded successfully
[INFO] - Starting trajectory generation...
```

If the log contains a **WandB run URL**, you can track progress in real time from the browser.

---

## Step 6: Make Your First Change

**Example: Add a new strategy**

```bash
# 1. Create strategy file
touch llm_tts/strategies/strategy_my_new.py

# 2. Implement your strategy (inherit from StrategyBase)

# 3. Create tests
mkdir tests/my_new
touch tests/my_new/test_my_new.py

# 4. Register in strategy registry
# Edit tests/strategy_registry.py

# 5. Validate
python tests/strategy_registry.py --validate

# 6. Run tests
pytest tests/my_new/ -v
```

See [Strategy Registration Guide](STRATEGY_REGISTRATION.md) for detailed steps.

---

## Step 7: Daily Development Workflow

```bash
# Make your changes...

# Format and check before committing
make fix     # Auto-fix with black, isort
make lint    # Check with flake8

# Run relevant tests
pytest tests/your_module/ -v

# Commit (hooks run automatically)
git commit -m "feat: add new feature"

# Push
git push origin your-branch
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit` and will:

- Format code with black and isort
- Check for trailing whitespace and large files
- Run flake8 linting
- Block commit if checks fail

To install or reinstall hooks:

```bash
make hooks
```

---

## Development Commands

```bash
# Testing
make test              # Run all tests
pytest tests/path/ -v  # Run specific tests

# Code Quality
make fix               # Auto-fix formatting (black, isort)
make format            # Format only (no other hooks)
make lint              # Check with flake8
make hooks             # Install pre-commit hooks

# Validation
python tests/strategy_registry.py --validate  # Validate all strategies
python tests/strategy_registry.py --list      # List registered strategies
```

---

## Pull Request Guidelines

1. Ensure all tests pass: `make test`
2. Run `make fix` and `make lint` before committing
3. Add tests for new strategies (register in `tests/strategy_registry.py`)
4. Keep PRs focused — one feature or fix per PR
5. Write clear commit messages describing what changed and why

---

## Troubleshooting

### vLLM fails with `RuntimeError: Engine core initialization failed`

Clear the vLLM compile cache and retry:

```bash
rm -rf ~/.cache/vllm/torch_compile_cache/
```

### Disk quota exceeded during model download

Configure `HF_HOME` to point to a larger disk (see Step 2 above).

### Tests skip with "API key not set"

Some tests require `OPENROUTER_API_KEY`. Set it in `.env` or export it directly:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```
