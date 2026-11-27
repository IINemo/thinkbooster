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
- Live MATH-500 run, BELT-H at threshold 0 (new samples, real stops, free
  answers): 86.4% (432/500) at 2,621 reasoning tokens (-17%), BELT stopped
  307 of 500 (61%). The offline replay predicted 86.6% at 2,576 (-18%), 62%.
  All output tokens: 3,014 vs 3,614 without stopping. Per problem: 419
  correct in both runs, 25 only without stopping, 13 only with BELT.
- vLLM backend: within bf16 noise of HF on fixed prefixes (closer to fp32 than
  HF on the long case); 99.2–99.5% identical decisions on 4,201 Minerva
  points; accuracy within one problem; 1.7x faster scoring per decision
  (2.2x at 4k–8k tokens), measured with cold prefills.

## Page review (2026-11-27)

An independent UX pass (desktop and phone, light and dark) and a content pass
for the supervisor version. Changes:

- Header: paper citation, a results-at-a-glance row and a one-paragraph bottom
  line that states where a plain token cap does as well.
- New "BELT inside ThinkBooster" section: the per-problem loop
  (VLLMStepGenerator, StepScorerBELT, BeltStoppingPolicy, StrategyBELT) and
  the Hydra config that wires it.
- New "Two backends" section: an HF vs vLLM table (mechanism, memory,
  requirements, fp32 error, decision agreement, Minerva replay at threshold 0,
  per-decision speed by prefix length). The Minerva replay with vLLM stop
  scores: 41.2% at 3,024 tokens vs HF 41.5% at 3,028 (one problem apart).
- The timeline now says its stop score is BELT-H (reads the probe's 768-d
  state), not a function of the four plotted beliefs; BELT-B and BELT-H are
  defined before the chart. Block, decision point, y_i, a_i and T are defined.
- The threshold slider shows its whole-dataset effect under the chart, and a
  synced copy sits on the accuracy chart.
- Accuracy table: grouped two-row header, consistent method names, a
  one-problem noise scale, saving guides at 20/30/40% on the chart, direct
  labels for all four curves. Result text names the columns where Now-only or
  the fixed cap win.
- Token price explained as accuracy points per 1,000 tokens (BELT's utility
  is accuracy minus lambda x tokens).
- Checks section moved before the results. Light-mode belief colours darkened
  to reach 3:1 on the surface; hint text uses the muted colour.
- Problem text and answers render with KaTeX; long questions keep their start
  and end and are cut only outside $...$. The exporter no longer truncates
  questions at 600 characters (it cut off the asked Minerva subproblem).
- Problem maths renders as MathML (KaTeX, output "mathml"): the artifact host
  blocks third-party stylesheets, so KaTeX's CSS and fonts cannot load.
- The page data JSON files were not in git (a repo-wide `*.json` ignore);
  they are now force-added.
- Each accuracy curve is now described on the page (what score is swept),
  with a measured explanation of why a token cap keeps up on MATH-500: the
  longest quarter of runs holds 65% of reasoning tokens and is right 70% of
  the time (96% below the median length). For a 30% saving the cap stops the
  longest 23% of runs and loses 26 problems; BELT-H stops 71% and loses 31.
- The cap explanation now shows, per run-length group, where each method
  stops at a 30% saving, and that BELT-H's peak stop score is typically -2.4
  on the longest runs vs +7.0 on short ones. A first look at "BELT-H score
  reaches the threshold or the run passes a cap": 86.4 / 84.6 / 82.8% at
  20 / 30 / 40% saving, above both parts; its two settings were chosen on the
  same runs, so it still needs a held-out check.
- Automated checks: all 772 trajectories at five thresholds, both frontiers
  and all 17 prices show no overlapping labels and no runtime errors.

## Open items

- The three end-to-end backend timing runs (bench_hf_b64, bench_hf_b9,
  bench_vllm_b9) are running; add their numbers to the backends table.
- GPQA-Diamond, AIME 2025 and HumanEval+ runs follow in the queue.
- Codex review is unavailable on the user's login (every model refused), so
  the branch has no Codex review yet; required before any PR.
