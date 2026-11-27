# BELT tutorial: development process

Goal: real examples of BELT (belief-based termination) running in ThinkBooster
for a NeurIPS tutorial, presented as an interactive page that explains the
mechanism (four beliefs over time) and reports our own numbers as they are.

Published page: https://claude.ai/artifact/Lti648KWJgRmAdcy1eoGHY (private;
share from the page's Share menu). Source: `index.html` plus the three JSON
files in this folder, exported by `scripts/belt_tutorial/export_artifact.py`.

## Pipeline

All code is in `scripts/belt_tutorial/`, configs in
`config/experiments/belt/tutorial/`. Runs live on MBZUAI-Artem-1 under
`/mnt/data/users/vlad.smirnov/belt_tutorial/` (GPU 1 only; GPU 0 there is stuck).

1. `make_policy.py`: BELT's published seed-0 policy fits
   (`examples/qwen3-1.7b-sample/policies` on BELT main) to ThinkBooster's
   policy format. Reproduces BELT's reference logits to 1e-11.
2. `run_queue.sh`: resumable per-GPU queue of `run_tts_eval.py` jobs. A run is
   done when `metrics.json` exists (`results.json` is a per-chunk checkpoint).
3. Never-stop runs (`natural` policy, threshold 1e9) log the four beliefs and
   the H policy logit at every decision point (every 4th block, blocks 4–128).
4. `forced_answers.py`: closes thinking at each decision point and asks for a
   boxed answer (48 tokens, brace-balanced extraction), graded like the runner
   (`pre_extracted=True`) in worker processes with a time limit.
5. `analyze.py`: label-free threshold calibration, offline replay of any
   threshold, run summaries.
6. Checks: `sample_parity.py` (reader vs the paper's archived scores),
   `backend_agreement.py` and `backend_speed.py` (HF vs vLLM feature backend),
   `tests/belt/validate_vllm_gpu.py` (fp32 noise-floor validation).
7. `export_artifact.py` -> page data.

## Decisions

- Mechanism first: the user chose "mechanism + our numbers" over reproducing
  the paper's 8-chain protocol (2026-11-27).
- Online BELT runs use threshold 0 (the fitted policy's own boundary). The
  label-free Minerva thresholds targeting 30/45/60% projected saving proved
  too aggressive (MATH-500 smoke: 94% -> 69% at the 45% one).
- HF scoring backend for all headline numbers; vLLM backend measured separately.

## Findings (2026-11-27)

- Integration is faithful: on the paper's 16 archived trajectories (492
  decision points) ThinkBooster's reader matches the archived beliefs
  (median diff 0.0015) and stop decisions (99.8%).
- Now belief AUROC for forced-answer correctness: 0.85 MATH-500, 0.75 Minerva.
- MATH-500 (500, one chain each, natural 88.8% at 3,148 reasoning tokens):
  BELT-H saves 1.3 / 13.3 / 17.8% at 0 / 1 / 2 points of accuracy loss and
  is best at a 20% saving; from about 30% saving a fixed token cap matches or
  beats it. The paper's ~41% OOD saving used 8 chains/question, integer-answer
  subsets and interpolated frontiers.
- Minerva Math (272, natural 44.1%): small gains; Now alone is uninformative.
- vLLM backend: within bf16 noise of HF on fixed prefixes (closer to fp32 than
  HF on the long case); 99.2–99.5% identical decisions on 4,201 Minerva
  points; accuracy within one problem; 1.7x faster scoring per decision
  (2.2x at 4k–8k tokens), measured with cold prefills.

## Open items

- MATH-500 online BELT run (threshold 0) and the three end-to-end backend
  timing runs are in the queue; add their numbers to the page.
- GPQA-Diamond, AIME 2025 and HumanEval+ runs follow in the queue.
- Codex review is unavailable on the user's login (every model refused), so
  the branch has no Codex review yet; required before any PR.
