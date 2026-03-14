# Experiment Results Visualizer

View experiment results from `run_tts_eval.py` in the Visual Debugger — without re-running strategies live.

## Quick Start

```bash
# 1. Run an experiment (tree data is now saved automatically)
python scripts/run_tts_eval.py --config-path=../config --config-name=experiments/beam_search/...

# 2. Convert results to debugger format
python scripts/convert_results_to_debugger.py outputs/<date>/<run_name>/ --install

# 3. Serve and open in browser
python -m http.server 8080 -d service_app
# Open http://localhost:8080/static/debugger/index.html
```

## Converter Options

```bash
# Basic: creates debugger_payload.json in the output dir
python scripts/convert_results_to_debugger.py outputs/<path>/

# Install as cached_examples.json (auto-loads in debugger)
python scripts/convert_results_to_debugger.py outputs/<path>/ --install

# Only incorrect samples (for debugging failures)
python scripts/convert_results_to_debugger.py outputs/<path>/ --incorrect-only

# Limit number of samples
python scripts/convert_results_to_debugger.py outputs/<path>/ --max-samples 50

# Custom output path
python scripts/convert_results_to_debugger.py outputs/<path>/ --out my_results.json
```

## Using the Debugger

1. Samples appear in the **Scenario** dropdown
2. Select a strategy/scorer and click **Run** to see the tree
3. **Timeline** (left) — click through reasoning steps
4. **Tree** (bottom) — orange path = selected, grey = pruned
5. **Candidates** panel — scores and text for each candidate at a step
6. **Prev/Next** buttons — navigate between samples
7. **Incorrect only** checkbox — filter to failed samples
8. **Load File** button — load a `debugger_payload.json` without `--install`

## What the Tree Shows

- Each node is a candidate generated at a reasoning step
- **Orange path**: the beam the strategy selected as its final answer
- **Grey nodes**: candidates that were generated and scored but pruned
- Click any node to see its full text and scores

## Changed Files

| File | Change |
|------|--------|
| `scripts/run_tts_eval.py` | Save tree data (`step_candidates`, `all_trajectories`, etc.) to `results.json` — previously discarded |
| `scripts/convert_results_to_debugger.py` | **New.** Converts experiment output to debugger JSON format |
| `service_app/static/debugger/index.html` | Added file upload input and sample navigation (Prev/Next, Incorrect only filter) |
| `service_app/static/debugger/app.js` | File upload handler, sample navigation logic, auto-enable cached mode for offline use |
| `config/experiments/beam_search/math500/beam_search_openrouter_gpt4o_mini_math500_entropy.yaml` | **New.** Quick-test config for OpenRouter beam search on MATH-500 |

## Supported Strategies

All strategies that produce tree data work:
- **Beam Search** — full tree with per-step candidates and beam lineage
- **Online Best-of-N** — stepwise candidate pools
- **Offline Best-of-N** — trajectory-level reranking with per-step breakdown
- **Self-Consistency** — parallel reasoning paths with voting

Baseline (single-pass) also works but shows a linear chain (no branching).

## Note on Old Experiments

Experiments run **before** the `run_tts_eval.py` change won't have tree data in `results.json`. The converter still works — it falls back to a stepwise view (one candidate per step) — but you won't see the full branching tree. Re-run the experiment to get tree data.
