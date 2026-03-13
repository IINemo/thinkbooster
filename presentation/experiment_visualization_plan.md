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

**IMPORTANT: `results.json` does NOT save the tree-building data.** The strategies return `step_candidates` (beam search, online BoN) and `all_trajectories` (offline BoN, self-consistency) in their result dicts, but `run_tts_eval.py` discards them (lines 1631-1674 cherry-pick only a subset of fields). Currently saved:
- `steps` — list of step dicts with `text`, `token_ids`, `generation_scores`, `other_data`
- `validity_scores` — flat list of per-step scores
- `generated_trajectory` — concatenated text
- `extracted_answer`, `answer_step`, `token_stats`, completion info

**Not saved (but available in strategy return value):**
- **Beam Search / Online BoN**: `step_candidates` — per-step decision points with all candidates, their scores, beam UIDs, parent linkage. This is the full tree structure.
- **Offline BoN**: `all_trajectories`, `all_scores`, `all_step_scores`, `best_idx` — all N candidate trajectories with scores.
- **Self-Consistency**: `all_trajectories` — all sampled paths.

---

## Proposed Approach

### Two things are needed:

**1. Save tree data in `run_tts_eval.py`** — modify `_generate_trajectories_batch()` (line 1631) to also save `step_candidates` and `all_trajectories` to results.json (or a separate `tree_data.json` to keep results.json lightweight).

**2. A standalone converter script** that reads the saved data and converts it to the debugger format:
- Reads `results.json` (with newly-saved tree fields) from an experiment output dir
- Calls the existing `convert_strategy_result_to_debugger_run()` for each sample
- Outputs a `cached_examples.json`-compatible file for the debugger to load

### Architecture

```
                          Step 0 (one-time)
                    ┌──────────────────────────┐
                    │ Modify run_tts_eval.py   │
                    │ to save step_candidates  │
                    │ and all_trajectories     │
                    └──────────────────────────┘

Experiment output dir          Converter              Visual Debugger
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ results.json     │────▶│ convert_results  │────▶│ Load as "cached  │
│  (with tree data)│     │ _to_debugger.py  │     │  example" or via │
│ metrics.json     │     │                  │     │  file:// protocol│
│ config.yaml      │     │ Uses existing    │     │                  │
│                  │     │ debugger_events  │     │                  │
└──────────────────┘     │ .py converter    │     └──────────────────┘
                         └──────────────────┘
```

---

## Implementation Steps

### Step 0: Save Tree Data in `run_tts_eval.py` (PREREQUISITE)

Currently `_generate_trajectories_batch()` at line 1631 builds `result_dict` without tree-building fields. Add:

```python
# After line 1660 in _generate_trajectories_batch()
# Save tree visualization data (if strategy provides it)
for key in ("step_candidates", "all_trajectories", "all_scores", "all_step_scores", "best_idx"):
    if key in result:
        result_dict[key] = result[key]
```

**Option A** — save directly in `results.json` (simpler, but increases file size significantly for beam search with many candidates).

**Option B** — save to a separate `tree_data.jsonl` file (one line per sample, keyed by index). Keeps `results.json` lightweight. The converter script would then read both files.

**Note on serialization:** `step_candidates` contains `StepCandidate` objects. These are already serialized as dicts with `text`, `token_ids`, `generation_scores`, `other_data` fields (same as `steps`), so JSON serialization should work. Verify with a test run.

### Step 1: Converter Script (`scripts/convert_results_to_debugger.py`)

**Input:** path to experiment output directory (containing `results.json` with tree data)
**Output:** JSON file in the debugger payload format

```python
# Pseudocode
from service_app.core.debugger_events import convert_strategy_result_to_debugger_run

def convert_experiment(output_dir, strategy_type, scorer_type=None):
    results = load_json(output_dir / "results.json")
    config = load_yaml(output_dir / ".hydra/config.yaml")  # experiment config

    examples = []
    for sample in results:
        # The result dict now contains step_candidates / all_trajectories
        # (saved by the modified run_tts_eval.py)
        run_payload = convert_strategy_result_to_debugger_run(
            strategy={"id": strategy_type, "name": ..., "family": ...},
            scorer={"id": scorer_type, ...} if scorer_type else None,
            strategy_result=sample,  # pass the full saved result
            budget=config.strategy.max_steps,
            latency_ms=0,
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

### Step 2: Adapt `debugger_events.py` for Serialized Data

The existing converter expects live `StepCandidate` objects (with `.text` attribute). Serialized results have dicts (with `"text"` key). Need to handle both:

- `_build_events_from_step_candidates()` — already works with dict-like candidates (check this)
- `_build_events_from_trajectory_pool()` — needs to accept step dicts instead of `StepCandidate` objects (access `.text` vs `["text"]`)
- `_build_stepwise_events()` — same: accept step dicts

**This is the main coding task** — make the converter accept both live objects and serialized JSON. A simple helper can bridge the gap:

```python
def _step_text(step):
    """Get text from either a StepCandidate object or a serialized dict."""
    return step.text if hasattr(step, "text") else step.get("text", str(step))
```

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

3. **Modify `run_tts_eval.py`** (Step 0) — add `step_candidates` and `all_trajectories` to the saved result dict. This is ~5 lines of code. Re-run one experiment to generate data with tree fields.

4. **Write the converter script** — `scripts/convert_results_to_debugger.py`:
   - Load `results.json` (with tree data) from experiment dir
   - For each sample, call the existing converter (or a thin wrapper)
   - Output a debugger-compatible JSON

5. **Handle serialization gap** — the converter expects `StepCandidate` objects but results.json has dicts. Create a lightweight adapter or modify the converter to accept both.

6. **Test with file:// protocol** — open `index.html` directly in a browser with the generated JSON as `cached_examples.json` in the same directory.

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

**Answer to the colleague's question:** You don't need to build trees inside `run_tts_eval.py`. The tree construction logic already exists in `debugger_events.py`. But there's a prerequisite: `run_tts_eval.py` currently **discards** the tree-building data (`step_candidates`, `all_trajectories`) when saving results. So the plan is:

1. **Modify `run_tts_eval.py`** (~5 lines) to also save `step_candidates` / `all_trajectories` to disk — this is the raw tree structure that strategies already compute but we throw away
2. A **post-hoc converter script** (`scripts/convert_results_to_debugger.py`) that reads experiment outputs and calls the existing `debugger_events.py` converter to produce the frontend-ready format
3. Minor **adaptation of `debugger_events.py`** to accept serialized dicts (from JSON) in addition to live `StepCandidate` objects
4. Optionally, a **"Load experiment" UI** in the debugger for convenience

The tree data is already computed by strategies at runtime — we just need to stop discarding it and then pipe it through the existing conversion layer.
