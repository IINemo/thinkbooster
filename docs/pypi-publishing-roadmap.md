# PyPI Package Publishing Roadmap

## Current state
- `pyproject.toml` has package metadata, `name = "thinkbooster"`, name is available on PyPI
- `pip install -e .` works locally but only because `setup.sh` installs extra deps

## Obstacles

### 1. `lm-polygraph` dependency
We use the `dev` branch (version `0.0.0`) which has features not in PyPI `0.5.0` (`VLLMWithUncertainty`, `VLLMLogprobsCalculator`, `api_with_uncertainty`). Options:
- Publish `lm-polygraph>=0.6.0` to PyPI with dev branch changes, then depend on it (cleanest)
- Add import guards and make it optional (works for API-only usage)

### 2. `latex2sympy2`
Installed with `--no-deps` to avoid antlr4 conflict with Hydra. Bundled copy exists at `llm_tts/evaluation/latex2sympy/` but isn't used by imports.

### 3. Hard imports without guards
~15 files import `lm_polygraph` at top level without `try/except`. Package crashes on `import llm_tts` without it.

### 4. `llm_tts/__init__.py` missing
Root package has no init file.

### 5. Package-data config wrong
`pyproject.toml` points to nonexistent `llm_tts/config/`. Actual configs at root `config/`. `service_app/static/` assets also not included.

### 6. Console script entry point broken
`run-tts-eval = "scripts.run_tts_eval:main"` won't work: `scripts/` not in package discovery, has relative imports.

### 7. Author metadata placeholder
`"Your Name"` / `"your.email@example.com"`.

## Steps to implement

- [ ] Publish `lm-polygraph>=0.6.0` to PyPI (or add import guards for all lm_polygraph imports)
- [ ] Create `llm_tts/__init__.py` with `__version__`
- [ ] Add `try/except ImportError` guards for `lm_polygraph` and `latex2sympy2` in all affected files
- [ ] Fix `pyproject.toml`: author metadata, package-data, remove broken console script
- [ ] Add `service_app` package-data for static files
- [ ] Create `MANIFEST.in` for sdist (config yamls, prompts, static assets)
- [ ] Test: `python -m build` → install wheel in clean venv → `import llm_tts` works
- [ ] Publish to PyPI with `twine upload`

## Files that need import guards

### `lm_polygraph`
- `llm_tts/models/blackboxmodel_with_streaming.py`
- `llm_tts/generators/api.py`
- `llm_tts/generators/huggingface.py`
- `llm_tts/scorers/step_scorer_prm.py`
- `llm_tts/scorers/estimator_uncertainty_pd.py`
- `llm_tts/evaluation/alignscore.py`
- `llm_tts/strategies/deepconf/strategy.py`
- `llm_tts/strategies/deepconf/utils.py`

### `latex2sympy2`
- `llm_tts/evaluation/parser.py`
- `llm_tts/evaluation/grader.py`
