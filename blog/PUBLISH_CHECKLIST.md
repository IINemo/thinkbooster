# Pre-publish checklist — ThinkBooster article

Internal notes for `ARTICLE.md`. Do not paste into Medium/LinkedIn.

## Before publishing
- [ ] **Figures/tables** — insert at the marked `> _[Figure: ...]_` spots; export `tab:gpt_oss` and `tab:ttc_frameworks` as images.
- [ ] **Live demo** — linked in the CTA (`http://demo-thinkbooster.nlpresearch.group`, verified up 2026-06-30). It is http-only and runs on an ephemeral RunPod pod, so keep the pod up through launch; serving https on the custom domain would be more robust.
- [ ] **K2 + ReProbe** — the MBPP+ number is single-seed (matched N=1 single-shot vs N=4 best-of-N, seed 42, 378-problem split; baseline = SLURM job 2651567, posted to PR #257). Replace with a multi-seed mean if available before launch.
- [ ] **OptiLLM** — the article no longer prints a specific OptiLLM count (softened to "many prompt and inference techniques" after codex review), so no number to verify. Optional: confirm their current count if you want to add one back.

## Verified this round (codex review, 2026-06-30)
- Service route names in the snippets match the live service: strategies `self_consistency` / `offline_bon` / `online_bon` / `beam_search`, scorers `entropy` / `perplexity` / `sequence_prob` / `prm`. (Fixed an invalid `best_of_n/confidence` route to `offline_bon/entropy`.)
- The `running_locally.md` link resolves on `main` (PR #259 merged).
- Library snippet (`BlackboxModelWithStreaming` / `StepCandidateGeneratorThroughAPI` / `StrategySelfConsistency` / `generate_trajectories_batch`) matches the current API.
- Numbers checked against the paper tables: HumanEval+ 79.3→88.8, CUDA 26→30 / −5pp syntax / 65→64 compile, beam+PRM 17–24× vs best-of-N+PRM. Math claim scoped (PRM/self-consistency win, not entropy); "visual debugger" reframed (ReasonGraph also has one); "first tool" and "20-plus for OptiLLM" superlatives removed.
- arXiv link (`2606.06915`) is live; the screencast resolves to the YouTube video.

## Notes
- `tts_metadata` field names: only verify if the extended metadata example is ever added (it is not currently in the article).
- ReProbe vs UHead: the public name is **ReProbe** (deliberate); the saved experiment branch/class still says `uhead`/`step_reasoning`.
