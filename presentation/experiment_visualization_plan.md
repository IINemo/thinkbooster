# Plan: Visualizing Experiment Results in the Visual Debugger

## Goal

Enable viewing experiment results (from `scripts/run_tts_eval.py`) in the existing Visual Debugger UI — without running strategies live.

---

## Current Architecture

### How the Debugger Works Now

1. **Backend** runs a strategy via `strategy_manager.py` → gets raw result dict
2. **`debugger_events.py`** converts the result dict into a list of **events** (the universal visualization format)
3. **`app.js`** receives events and builds an interactive tree via `buildTreeFromEvents(events)`

The key conversion layer is `debugger_events.py:convert_strategy_result_to_debugger_run()` — it transforms raw strategy output into the event format the frontend understands.

### Event Format (what the frontend needs)

```json
{
  "step": 1,
  "title": "Step 1: Candidate generation",
  "stage": "tree_expand",
  "signals": [{"name": "confidence", "value": 0.85, "direction": "higher_better"}],
  "candidates": [
    {
      "id": "step_1_candidate_0",
      "label": "Candidate 1",
      "text": "Let me think about this...",
      "status": "selected",
      "selected": true,
      "signals": {"confidence": 0.85, "prm": 0.92},
      "beam_uid": 1,
      "parent_beam_uid": null
    }
  ]
}
```

### What `run_tts_eval.py` Already Saves

| File | Content |
|------|---------|
| `results.json` | Per-sample results: steps, scores, trajectory, extracted_answer |
| `candidates.json` | Multi-trajectory data (offline BoN) |
| `sample_metrics.jsonl` | Per-sample compute metrics |
| `metrics.json` | Aggregated accuracy, tokens, etc. |

**Crucially, `results.json` already contains the raw data needed for tree building:**
- **Beam Search / Online BoN**: `step_candidates` field with candidates, scores, beam UIDs, parent linkage
- **Offline BoN**: `all_trajectories`, `all_scores`, `all_step_scores`, `best_idx`
- **Self-Consistency**: `all_trajectories` with consensus info
- **Baseline**: `steps` + `validity_scores`

---

## Proposed Approach

### No separate script needed inside `run_tts_eval.py`

The conversion from raw results → debugger events already exists in `debugger_events.py`. We just need a **standalone converter script** that:

1. Reads `results.json` (and optionally `candidates.json`) from an experiment output dir
2. Calls the existing `convert_strategy_result_to_debugger_run()` for each sample
3. Outputs a `cached_examples.json`-compatible file (or a new format for the debugger to load)

### Architecture

```
Experiment output dir          Converter              Visual Debugger
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ results.json     │────▶│ convert_results  │────▶│ Load as "cached  │
│ candidates.json  │     │ _to_debugger.py  │     │  example" or via │
│ metrics.json     │     │                  │     │  file:// protocol│
│ config.yaml      │     │ Uses existing    │     │                  │
│                  │     │ debugger_events  │     │                  │
└──────────────────┘     │ .py converter    │     └──────────────────┘
                         └──────────────────┘
```

---

## Implementation Steps

### Step 1: Converter Script (`scripts/convert_results_to_debugger.py`)

**Input:** path to experiment output directory (containing `results.json`)
**Output:** JSON file in the debugger payload format

```python
# Pseudocode
from service_app.core.debugger_events import convert_strategy_result_to_debugger_run

def convert_experiment(output_dir, strategy_type, scorer_type=None):
    results = load_json(output_dir / "results.json")
    config = load_yaml(output_dir / ".hydra/config.yaml")  # experiment config

    examples = []
    for sample in results:
        # Reconstruct the result dict format that debugger_events expects
        raw_result = reconstruct_strategy_result(sample, strategy_type)

        # Use existing converter
        run_payload = convert_strategy_result_to_debugger_run(
            result=raw_result,
            strategy_id=strategy_type,
            scorer_id=scorer_type,
            ...
        )

        examples.append({
            "id": f"sample_{sample['index']}",
            "title": sample["question"][:80],
            "description": f"Gold: {sample['gold_answer']}, Predicted: {sample['extracted_answer']}",
            "payloads": {"default": make_payload(run_payload, sample)}
        })

    save_json(examples, output_dir / "debugger_payload.json")
```

### Step 2: Adapt `debugger_events.py` for Offline Data

The existing converter expects live strategy result objects (with `StepCandidate` instances). Experiment results have serialized data (text strings instead of objects). Need minor adaptation:

- `_build_events_from_step_candidates()` — already works with dict-like candidates
- `_build_events_from_trajectory_pool()` — needs to accept serialized trajectories (strings instead of StepCandidate objects)
- `_build_stepwise_events()` — needs to accept step text strings

**This is the main coding task** — make the converter accept both live objects and serialized JSON.

### Step 3: Add "Load from File" to the Debugger UI

Two options (pick one):

**Option A (simpler):** Generate a `cached_examples.json` and open the debugger HTML as `file://` — already supported, no backend needed.

**Option B (richer):** Add a "Load experiment" button to the debugger that accepts a JSON file upload or a directory path. This would:
- Add a file input element in `app.js`
- Parse the uploaded JSON into the same format as cached examples
- Populate the example selector dropdown

### Step 4: Multi-Sample Navigation

Current debugger shows one problem at a time. For experiments with hundreds of samples, add:
- Sample index selector (dropdown or prev/next buttons)
- Filter by correctness (show only incorrect samples for debugging)
- Summary stats bar (accuracy, avg tokens)

---

## Where to Start

### For the colleague — recommended order:

1. **Start with `service_app/core/debugger_events.py`** — understand `convert_strategy_result_to_debugger_run()` (line 49). This is the core function. Read its input/output contract.

2. **Read one cached example** — look at `service_app/static/debugger/cached_examples.json` to see the exact output format the frontend expects. Focus on `strategies[].run.events[]`.

3. **Write the converter script** — `scripts/convert_results_to_debugger.py`:
   - Load `results.json` from experiment dir
   - For each sample, call the existing converter (or a thin wrapper)
   - Output a debugger-compatible JSON

4. **Handle serialization gap** — the converter expects `StepCandidate` objects but results.json has plain strings. Create a lightweight adapter or modify the converter to accept both.

5. **Test with file:// protocol** — open `index.html` directly in a browser with the generated JSON as `cached_examples.json` in the same directory.

---

## Key Files to Read

| File | Why |
|------|-----|
| `service_app/core/debugger_events.py` | **Core converter** — strategy result → events |
| `service_app/static/debugger/cached_examples.json` | **Target format** — what the frontend expects |
| `service_app/static/debugger/app.js:2052-2256` | `buildTreeFromEvents()` — how frontend builds the tree |
| `service_app/core/visual_debugger_demo.py` | How demo payloads are assembled |
| `scripts/run_tts_eval.py:1630-1674` | What fields are saved per sample in `results.json` |

---

## Summary

**Answer to the colleague's question:** No, you don't need a separate script running inside `run_tts_eval.py`. The conversion layer already exists in `debugger_events.py`. What you need is:

1. A **post-hoc converter script** (`scripts/convert_results_to_debugger.py`) that reads experiment outputs and calls the existing converter
2. Minor **adaptation of `debugger_events.py`** to accept serialized (JSON) data instead of only live Python objects
3. Optionally, a **"Load experiment" UI** in the debugger for convenience

The data is already there — `step_candidates` (for beam/online BoN) and `all_trajectories` (for offline BoN/SC) contain everything needed for tree visualization. The gap is only in the serialization format.
