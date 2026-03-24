# Analysis: Beam Finalization Logic — `completed_beams or active` vs `completed_beams + active`

## The Code in Question

```python
# Line 703 in strategy_beam_search.py
# This runs AFTER the main loop ends (max_steps reached, sample still in active_samples)

for sample_id in active_samples:
    active = sample_beams[sample_id]
    candidates = completed_beams_by_sample[sample_id] or active   # ← THIS LINE
    best_beam = self._select_best_beam(candidates)
```

## How Beam Search Works (Two Exit Paths)

### Exit Path 1: All beams complete inside the loop (line 643)
A sample's beams are split every step into `completed` and `active` lists. Completed beams are stored in `completed_beams_by_sample` and removed from the active pool. When `active` becomes empty (all beams finished), `_select_best_beam(completed_beams_by_sample[sample_id])` is called — this only considers completed beams. **No issue here.**

### Exit Path 2: `max_steps` reached with beams still active (line 699–704)
The main `for step_num in range(max_steps)` loop ends. Some samples may still have active beams. This is where line 703 runs. At this point:
- `completed_beams_by_sample[sample_id]` = beams that finished early (EOS, `is_trajectory_complete`, etc.)
- `sample_beams[sample_id]` = beams still active when max_steps was reached (never produced EOS)

## What `or` Does

Python `or` on lists: `[B1, B2] or [A1, A2, A3]` → `[B1, B2]` (first non-empty wins).

| `completed_beams` | `active` | `candidates` (with `or`) |
|---|---|---|
| `[]` | `[A1, A2, A3]` | `[A1, A2, A3]` — falls through to active ✅ |
| `[B1, B2]` | `[A1, A2, A3]` | `[B1, B2]` — **active beams ignored entirely** |
| `[B1]` | `[]` | `[B1]` ✅ |

## What `+` Would Do

`[B1, B2] + [A1, A2, A3]` → `[B1, B2, A1, A2, A3]` — all beams considered, `_select_best_beam` picks highest aggregated score.

## Concrete Examples

### Example 1: `or` causes suboptimal selection

```
beam_width=4, max_steps=30

Step 12: Beam B completes early (model emits EOS), score=0.65
         → moved to completed_beams
Steps 13–30: Beams A, C, D continue expanding
         A accumulates score=0.89, C=0.72, D=0.61

At max_steps:
  completed_beams = [B (0.65)]  ← non-empty
  active = [A (0.89), C (0.72), D (0.61)]

  With `or`:  candidates = [B (0.65)] → picks B (0.65) ❌
  With `+`:   candidates = [B, A, C, D] → picks A (0.89) ✅
```

Here `or` loses a much better beam.

### Example 2: `or` is actually correct

```
Game of 24, max_steps=5

Step 3: Beam B finds "= 24", completes, score=0.70
Steps 4–5: Beams A, C generate more steps but never reach "= 24"
           A has score=0.85 but trajectory is incomplete (no final answer)

At max_steps:
  completed_beams = [B (0.70, has answer "= 24")]
  active = [A (0.85, no answer), C (0.60, no answer)]

  With `or`:  candidates = [B (0.70)] → picks B with valid answer ✅
  With `+`:   candidates = [B, A, C] → picks A (0.85) but A has NO answer ❌
```

Here `or` correctly prefers the beam that actually solved the problem.

### Example 3: Mixed — both have answers

```
Step 15: Beam B completes with answer, score=0.65
Step 30 (max_steps): Beam A is still "active" but its last step also contains
                      an answer (answer_pattern detected but is_trajectory_complete=False
                      due to how the detector works)

  With `or`:  picks B (0.65), misses A which has a better answer
  With `+`:   picks A (0.85), gets the better answer
```

## The Core Question

**What does "completed" mean semantically?**

A beam is marked `completed` when `is_trajectory_complete=True` or `is_thinking_complete=True`. This is a **signal from the model** that it considers itself done (emitted EOS or stop token).

An "active" beam at max_steps is one where **the model wanted to keep going but we cut it off**. Its trajectory may or may not contain a usable answer depending on the task.

## My Assessment

**Neither `or` nor `+` is universally correct.** They encode different assumptions:

| Variant | Assumption | Good for | Bad for |
|---|---|---|---|
| `or` (current) | Completed beams are inherently better because they contain a full answer | Short tasks (Game of 24), tasks where answer completeness matters | Long math reasoning where max_steps is the normal exit |
| `+` (proposed) | Score is the best indicator of quality regardless of completion status | Long math reasoning where all beams hit max_steps | Tasks where active beams have no usable answer |

## Recommendation

The safest fix is neither `or` nor `+` — it's **`completed_beams + active` but with a completion bonus or penalty**:

```python
# Option A: simple — just combine and trust the scores
candidates = completed_beams_by_sample[sample_id] + active

# Option B: prefer completed but don't ignore active entirely
# (add a small bonus to completed beams' scores in _select_best_beam)

# Option C: task-aware — check if active beams have answers
candidates = completed_beams_by_sample[sample_id] + [
    b for b in active if self._has_answer_content(b["steps"][-1])
]
if not candidates:
    candidates = completed_beams_by_sample[sample_id] or active
```

**For now, I recommend Option A (`+`)** because:
1. In practice, most math benchmarks run all beams to max_steps (bug doesn't trigger with `or` anyway)
2. When it does trigger, `+` gives the better result in the common case (Example 1)
3. For Game of 24 / ToT verification, we should separately validate that completed beams actually have valid answers — that's an evaluator concern, not a beam selection concern
4. The scorer's job is to assign scores that reflect answer quality — we should trust it

But this should be tested on Game of 24 specifically to confirm.
