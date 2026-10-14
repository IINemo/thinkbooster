# ThinkBooster article — proposed structure (Medium first, LinkedIn cut after)

> **Full draft now written to [`ARTICLE.md`](ARTICLE.md).** Added a dedicated section,
> "ReProbe, on a model the paper never tested," presenting the K2-Think-V2 (MBZUAI + G42)
> + ReProbe best-of-N result (MBPP+ 66.1→66.9 plus / 80.7→81.7 base), framed as a
> model-agnostic-generality point, not an accuracy headline (single-seed; scaling up).
> ReProbe = *Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States
> of Large Language Models* — the group's own scorer, also one of the four scorer families.

Draft plan, revised after an independent fact-check pass. Not committed. Source of truth
for facts: the ACL 2026 paper (`_ACL_2026__ThinkBooster_demo (1)/latex/main.tex`), the
repo README, and `tables/ttc_frameworks.tex`. Numbers cross-checked against
`tables/qwen3_full_results.tex`, `qwen25_full_results.tex`, `gpt_oss.tex`.

---

## The thesis (one sentence)

Your model can already think longer; the open question is *which* way of spending that
test-time compute is cheapest for the accuracy you need — and ThinkBooster is the first
tool that lets you measure the answer and ship it by changing one URL.

This is the spine. Every section either sets it up, proves it, or acts on it. If a
paragraph doesn't serve the thesis, it gets cut.

## Audience

Two readers at once: the researcher (wants a reproducible, compute-aware benchmark and the
uncertainty scorers) and the builder (wants a drop-in proxy with a compute budget). The
piece speaks to the builder first, with enough rigor to keep the researcher.

## Title options

1. *Your model can think longer. ThinkBooster helps you decide how.*
2. *Test-time compute, without the guesswork.*
3. *One URL turns any LLM into its own "Pro reasoning mode."*

Lead candidate: #1 as title, #3 as subtitle.

---

## Section-by-section

### 0. Title, subtitle, hero image
- Hero: the framework diagram (`images/thinkbooster.png`) or the gateway diagram
  (`images/endpoint.pdf`).

### 1. The problem, in one scene  (~160 words)
- Thinking models (o1, R1, Qwen3, GPT-OSS) made one idea mainstream: spend more compute at
  inference, get better answers. **Add one plain primer sentence:** test-time compute
  scaling means paying more at answer time, not training time, to get a better answer.
- But there are many ways to spend it — sample and rerank, search a tree of steps, stop when
  the model is confident, add compute only on the hard steps. Each lives in its own repo, is
  measured its own way, and most never report what it costs.
- So the question a practitioner actually has has no clean answer: *for the accuracy I need,
  which method is cheapest?*
- Land the thesis. One line: ThinkBooster is built to answer that, and to let you switch the
  answer on without rewriting your app.
- Asset: hero diagram.

### 2. What ThinkBooster is  (~210 words)
- Plain definition, no adjectives: an open-source (MIT) framework that puts nine test-time
  scaling strategies and four scorer families behind one API, measures accuracy and compute
  together, and ships as an OpenAI-compatible proxy plus a visual debugger.
- The one design idea worth explaining: strategy and scorer are separate. The *strategy*
  decides the shape of the search (how many samples, tree or line, when to stop). The
  *scorer* decides which partial answer looks best. ThinkBooster ships four kinds, and naming
  them sets up the results section:
  - a process reward model (a separate critic model trained to grade reasoning steps),
  - the model's own uncertainty / confidence (read off its token probabilities, no extra model),
  - an LLM-as-judge,
  - ReProbes (a light probe on the model's internal states; the group's own work).
- You mix and match any strategy with any scorer. The four system pieces: library, benchmark,
  gateway, debugger.
- Asset: strategy table (the nine strategies).

### 3. The hook, in one line of code  (~110 words + one annotated line)
- The drop-in proxy. Point an OpenAI client's `base_url` at `.../v1/<strategy>/<scorer>` and
  the same model now does test-time scaling. No other code changes.
- Show only the annotated URL line here (the *concept*); the full runnable snippet lives in
  §6 so we don't print the same code twice.
- Why it matters: agents, copilots, and enterprise stacks already speak OpenAI. You add
  reasoning scaling by changing a string, and you keep the compute budget in your hands.
- Asset: gateway diagram (`images/endpoint.pdf`).

### 4. Does it actually work? (the results that surprised us)  (~280 words)
Write this as three short **paragraphs** with plain topic sentences — not a bolded-bullet
triad (that pattern is a known AI tell, and three parallel one-liners is rule-of-three).

- **Uncertainty as a near-free scorer (coding).** On HumanEval+, the MUR strategy with a plain
  entropy signal takes Qwen3-8B from 79.3 to 88.8 — the best coding number in the study, and
  higher than any reward-model setup. The point is the scorer's cost: entropy is read off the
  tokens the model already produced, so it adds almost nothing, whereas a reward model is a
  second network you have to run. Against the *most accurate* reward-model setup (beam search
  + PRM) MUR+entropy is both better and under a quarter of the compute. State the honest
  scope: on math a real PRM still wins, so this is a coding finding, not a universal one.
- **Spending more does not reliably buy more.** With the cheap scorers, beam search
  underperforms plain Best-of-N and even self-consistency, despite searching far harder. Its
  one strong configuration, beam search + PRM, does take the top math accuracy — but it costs
  17–24× more than Best-of-N + PRM to get there, often for a tie or a point or two.
  Best-of-N + PRM is the better accuracy-per-FLOP point for most budgets. (Do **not** say
  "beam loses to BoN" and "17–24×" in the same breath — they are different comparisons.)
- **A real engineering payoff.** On GPT-OSS-120B generating CUDA kernels, dropping in
  Best-of-N + PRM raised end-to-end correctness from 26 to 30 and cut syntax errors by 5
  points. Add the honest caveat the table shows: compilation rate dips a hair (65→64) because
  the reward model favors more ambitious kernels that are likelier to fail to compile — a
  trade the correctness gain pays for.
- Closing line that ties to the thesis: none of this is visible from an accuracy column
  alone; because the benchmark logs TFLOPs and tokens next to accuracy, you can see the trade
  and pick the point you can afford.
- Asset: the accuracy-vs-compute plot (`qwen3_humaneval_ratio.pdf`,
  `qwen25_aggregate_ratio.pdf`) and the GPT-OSS results table.

### 5. Which one should you use?  (~90 words, a shareable decision box)
A three-line rule of thumb straight off the results and the white/black-box split:
- Closed API, no access to logits: Best-of-N, majority voting, or extended thinking.
- Self-hosted and cost-sensitive: MUR or Best-of-N with an uncertainty scorer (free to
  compute, no second model).
- Need the most accuracy and can pay for it: beam search + PRM.
This is likely the most-shared element and it directly serves the thesis. Set it as a boxed
callout.

### 6. How it compares — vs OptiLLM and the rest  (~260 words + table)
- Honest opening: there are good tools here. OptiLLM is the closest — an OpenAI-compatible
  proxy with a wide menu of inference techniques. On raw count of tricks it has more than us;
  say so.
- The fairness note, kept *next to the table*: ThinkBooster's own paper claims its nine
  strategies "cover more than twenty" recent methods, and OptiLLM's full menu is "20+" too —
  so under a loose count both are 20+. Under a strict rule (genuine search/scoring/decoding
  methods, excluding prompt scaffolds) it is 9 vs 7. Use that symmetry; don't print "9 vs 7"
  next to an unexplained "20+".
- What we add that none of them ship:
  - uncertainty / confidence as a first-class, swappable scorer family (built on
    LM-Polygraph) — narrow wording on purpose (a rival ships entropy decoding; the claim is
    "first-class swappable family," not "nobody else touches confidence"),
  - process-reward-model scoring and FLOPs-level compute accounting (OptiLLM has neither),
  - a joint performance–compute benchmark with bundled, judged datasets (5 math, 3 coding,
    1 science),
  - a visual debugger for reasoning trajectories.
- Trim the per-competitor tour to the two names a reader knows (LLM Reasoners, OptiLLM); let
  the table carry OpenR / search-and-learn / TreeQuest / ReasonGraph.
- Honesty caveat in the caption: the cells are our own assessment, not an audited score.
  (Keep this — it is more candid than the paper itself, and it is the best anti-spin move.)
- Asset: the `ttc_frameworks` comparison table.

### 7. Try it in five minutes (usage + code)  (~230 words + snippets)
- Install: `pip install thinkbooster`.
- As a proxy: the full `base_url`-swap snippet with `extra_body` budget controls (use the
  exact model id from the README, `Qwen/Qwen3-30B-A3B`).
- As a library: import a strategy and run it (for researchers).
- Run the compute-aware benchmark: the `run_tts_eval.py` command that prints accuracy and
  TFLOPs.
- Open the visual debugger at `localhost:8001/debugger`.
- Optional: the response-metadata example (per-step scores, token count, TFLOPs) — flag for
  verification against the shipped API before publishing.
- Asset: debugger screenshot (`images/demo-treeview.png` or `demo-result.png`).

### 8. The honest limitations  (~140 words)
- White-box strategies (MUR, DeepConf, uncertainty CoT) need logits or hidden states, so they
  want open-weight or self-hosted models. Closed APIs get the black-box subset (Best-of-N,
  majority voting, extended thinking, LLM-as-judge).
- Step-boundary detection is still hard for native, unstructured thinking traces.
- Scope so far: math, coding, science QA. Compute is theoretical TFLOPs plus tokens;
  wall-clock is logged but not yet studied.
- Why this section exists: it is true, it sets correct expectations, and it is what keeps the
  piece from reading as a pitch.

### 9. Why you should try it / call to action  (~130 words)
- Restate the thesis: the question is not whether to scale test-time compute but which way
  and at what cost — and ThinkBooster lets you answer that and ship the answer with a URL
  change.
- Who it is for, one line each: researchers (reproducible compute-aware benchmark,
  uncertainty scorers) and builders (drop-in proxy, budget control).
- Trim to two asks: `pip install thinkbooster` and try the live demo. Links to GitHub /
  arXiv / paper go inline, not as a four-item list.

---

## Figure → section map (the user inserts these)
- §0 hero — `thinkbooster.png` or `endpoint.pdf`
- §2 — `tab:tts_strategies` (nine strategies) or `thinkbooster_library.pdf`
- §3 — `endpoint.pdf` (gateway)
- §4 — `qwen3_humaneval_ratio.pdf` + `qwen25_aggregate_ratio.pdf` + GPT-OSS table
- §6 — `ttc_frameworks` table
- §7 — `demo-treeview.png` / `demo-result.png` / `demo-config.png`
- (spare: `beam_search.pdf` to illustrate a strategy in §2 or §4)

## Length
- Medium: ~2,000 words, an 8–9 minute read. Comprehensive but not padded. If it runs long,
  cut the per-competitor one-liners in §6 first.
- LinkedIn 3-minute cut (~300 words): keep §1 hook, one line of §2, the §4 HumanEval result,
  the §3 URL line, the §5 decision box, and the §9 CTA. Drop the comparison detail and
  limitations (link to the Medium piece for those).

## Voice guardrails (so it does not read as AI-generated)
- Sentence-case headings. Sparse bold. Few em dashes (commas, periods, parentheses instead).
- Banned words: delve, leverage, seamless, robust, powerful, state-of-the-art, pivotal,
  underscore, showcase, testament, crucial, realm, landscape (figurative).
- Banned openers: sentence-initial "Additionally,", "Moreover,", "Furthermore," (the top
  AI-vocab tell).
- Banned shapes: "not just X but Y" / "X is not Y, it's Z" parallelisms; rule-of-three
  padding; bolded-header-colon bullet lists (write §4 as paragraphs); `-ing` tails that
  editorialize ("highlighting its importance").
- Concrete nouns and real numbers over adjectives. Plain "is/are", not "serves as". State
  affiliations flat ("LM-Polygraph, our group's UQ library"), not as self-importance
  ("the group's research edge / lineage").
- One clear argument, not a feature list with a bow on top.

## Claims to keep exactly (verified against the tables)
- HumanEval+ 79.3 → 88.8 (+9.5), MUR + entropy, Qwen3-8B.
- GPT-OSS-120B CUDA: correctness 26 → 30 (+4pp), syntax errors −5pp, compilation 65 → 64.
- Beam+PRM vs BoN+PRM on Qwen2.5 math = 17–24× compute, beam ties/wins on accuracy.
- Nine strategies, four scorer families. base_url `.../v1/<strategy>/<scorer>`. MIT license.

## Claims to fix or qualify (the two a competitor would unpick)
- Beam-search §4: do not weld "loses to BoN" to "17–24×". Separate the cheap-scorer
  underperformance from the beam+PRM cost premium.
- Uncertainty §4: "fraction of the compute" is true only vs the *strongest* PRM setup
  (beam+PRM). The durable claim is the near-zero *scorer* overhead, not that the strategy is
  globally cheaper (vs cheap BoN+PRM it is ~6× more). Keep it coding-scoped.
- §6: "only framework with uncertainty scorers" → "only one with uncertainty as a
  first-class, swappable scorer family."

## Open checks before publishing (facts to confirm)
- Is the live demo URL up? (reviewers flagged it down during review)
- `tts_metadata` field names match the shipped service?
- OptiLLM's current technique count (so the loose "20+" is accurate).
- Title wording: paper `\title` says "Seamless Test-Time Scaling"; body says "test-time
  compute scaling" — pick one.
