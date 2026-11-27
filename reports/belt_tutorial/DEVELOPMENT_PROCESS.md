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
- Held-out check of the combined rule (`validate_hybrid.py`, 500 random half
  splits, settings picked on one half and scored on the other;
  `hybrid_validation.json`). Keeping BELT-H at its fitted threshold 0 and
  tuning only a cap is as good as tuning both. MATH-500 held-out: 85.7 /
  83.3 / 81.8% at 20 / 30 / 40% saving vs cap 84.5 / 82.5 / 79.5 and BELT-H
  86.0 / 82.2 / 74.3. Minerva: ties the cap. The page shows these held-out
  numbers instead of the in-sample ones.
- Automated checks: all 772 trajectories at five thresholds, both frontiers
  and all 17 prices show no overlapping labels and no runtime errors.

## Deployment (2026-11-27)

- `thinkbooster/belt/engine.py` (`BeltEngine`): Qwen3-1.7B in vLLM with the
  BELT probe and a policy, loaded once; a stop closes `</think>` so the saved
  reasoning is never generated. Both feature backends; per-call threshold.
- `service_app`: `tts_strategy="belt"` (`BELT_POLICY_PATH`, optional
  `tts_belt_threshold`); the reply is the final answer, `tts_metadata.belt`
  reports thinking tokens and the stop decision.
- `examples/belt_deploy_vllm.py`: same problems with and without early
  stopping on one engine. Smoke test, 8 MATH-500 problems: HF backend 8/8
  both ways at 1,314 vs 2,105 thinking tokens; vLLM backend 7/8 vs 8/8 at
  -24%. API call: 966 vs 2,407 thinking tokens, same answer.
- End-to-end backend timing, 64 MATH-500 problems: HF batch 64 8.4 min, HF
  batch 9 12.6 min, vLLM batch 9 15.2 min (belief queries 7 ms each, but
  eager-mode generation and the batch cap make the run slower).
- Larger-model plan: `reports/belt_qwen36/PLAN.md` (Qwen3.6-35B-A3B).

## Qwen3.6-35B-A3B probe (2026-11-27)

- BELT changes live as a patch, `scripts/belt_qwen36/apply_belt_patch.py`
  (contract, text-only vLLM, fallback, Minerva screening, truncated bf16
  trainer backbone, reader memory fixes), applied to BELT 700948b on Artem-3.
- Pool built (7,100 questions, salt `qwen36-belt-2026-11-27-v1`), smoke test
  passed end to end; details and the memory fixes are in PLAN.md.
- Full fit-split generation: 16 resumable shards (`run_fit_queue.sh`), 4 per
  GPU on Artem-3 (CUDA 13 venv, since 19:27) and Artem-8 (since 20:31), about
  3.7 h per shard, done around 11:30 on 2026-11-28. Artem-1 (user quota full),
  Artem-2 and Artem-7 (taken by another user's jobs) dropped out.
- Artem-8's driver supports CUDA 12.9 only, so it runs a cu129 build of the
  same venv (`build_venv_cu129.sh`: torch 2.13.0+cu129, vLLM 0.31.0+cu129,
  everything else pinned to Artem-3's versions, no torchcodec) and samples
  with vLLM's own top-k/top-p kernels (`VLLM_USE_FLASHINFER_SAMPLER=0`; its
  system nvcc is too old for FlashInfer). Same sampling distribution; record
  the host per shard when merging.
- Data audit (`audit_chains.py`) on the smoke chains found that Qwen3.6
  writes about 8% of its paragraph breaks as indented blank lines
  (`"\n    \n"`). BELT's block separator counts them, but its vLLM stop string
  `"\n\n"` never stopped on them, so those block ends and their probes were
  lost (one 16k chain: 3 of 56 block ends recorded, no probes). BELT_v2 adds
  the indented forms as stop strings. Replaying the 32 smoke chains through
  the generator with a recorded-token engine (`replay_block_stops.py`)
  reproduces v1's block ends exactly and gives v2 block ends that match the
  text in 32 of 32 chains; probes go from 241 to 262.
- Shards 0, 1, 6, 7 started on v1; `repair_block_stops.py` recomputes their
  block ends by replay and generates only the missing forced answers with
  BELT's own request builder, seed keys and grader (`fit/shard<N>_v2`). All
  other shards run on BELT_v2.
- Not a bug: forced answers parse less often deeper in a chain (92% to about
  70%); the failures stop at `}` after 5 to 8 tokens, i.e. symbolic guesses
  such as `\frac{1`, which BELT's strict integer Now label counts as wrong.
  Length-capped chains are not repetition loops (repeated 20-grams in the
  last 3k tokens: median 1.5%, none above 50%).

## Frontier chart redrawn (2026-11-27)

A colleague could not read the accuracy-against-tokens chart: absolute tokens
on x put "no early stopping" at the far right, and the −20/30/40% guides had
no stated reference. It now plots the share of reasoning tokens saved (0% at
the left) against accuracy change in points (0 at the top), so higher is
better at every saving; x runs to 40% (the table's last column) unless the
threshold point is further out. A hover or tap readout lists all four rules
at one saving and bolds the best; a "How to read it" paragraph sits above.

## Open items

- The vLLM backend needs CUDA graphs and a larger batch before it can speed
  up whole runs.
- GPQA-Diamond, AIME 2025 and HumanEval+ runs follow in the queue.
- Codex review is unavailable on the user's login (every model refused), so
  the branch has no Codex review yet; required before any PR.
