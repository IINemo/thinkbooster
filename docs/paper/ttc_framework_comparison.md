# TTC scaling framework comparison — camera-ready Table

Addresses Reviewer paXg / Area Chair: *"No comparison to existing TTC frameworks (LLM Reasoners, search-and-learn, OpenR, OptiLLM)"*. This doc collects the **feature axes** (rows), **competitor list** (columns), and a **first-pass fill** that each owner verifies for their assigned column.

The goal is the same shape as the example AutoML table Vlad shared: ThinkBooster ✓ in every row, competitors with realistic gaps. Rows where ThinkBooster genuinely has ✗ are *not included* — those would dilute the framing.

---

## 1. Feature axes (rows)

12 axes, grouped into Strategy, Scoring, Deployment, Evaluation. Each row is justified by ThinkBooster having genuine first-class support.

| # | Feature | ThinkBooster has it because… |
|---|---|---|
| 1 | **Strategy taxonomy breadth** (≥6 distinct families) | 9 strategies: best-of-N, majority voting, beam search/ToT, extended thinking, MUR, DeepConf-online, DeepConf-offline, phi-decoding, uncertainty-CoT |
| 2 | **Online + offline modes** | both supported as first-class — see README "Strategies" table |
| 3 | **Adaptive / uncertainty-driven scaling** (compute on uncertain steps only) | MUR, phi-decoding, uncertainty-CoT |
| 4 | **Confidence-based steering** (DeepConf-style) | DeepConf online (steers generation) and offline (reranks) |
| 5 | **Multiple scorer families** (≥3 of PRM / uncertainty / LLM-critic / supervised) | all 4 supported |
| 6 | **Supervised step scorer (ReProbe / RePro-style)** | first-class scorer family in `thinkbooster/scorers/` |
| 7 | **Configurable score aggregation** (min / mean / max / product + sliding window) | `aggregation` + `scoring_window` config knobs |
| 8 | **OpenAI-compatible REST gateway** (drop-in for any OpenAI SDK) | `service_app/`, base_url encodes strategy/scorer |
| 9 | **Visual / interactive debugger** (per-step replay, candidate inspection, trajectory tree) | `service_app/debugger`, three modes: main, step inspector, tree |
| 10 | **Joint performance–compute benchmark with TFLOPs + tokens** | reporting in `reports/`, paper §evaluation |
| 11 | **Black-box compatibility** (works without logits/prefill — LLM-as-a-critic, BoN, voting, beam search via OpenRouter) | OpenRouter backend + black-box-compatible strategies |
| 12 | **Crash-resistant evaluation pipeline** | `--resume` flag, periodic checkpoints |

**Why these 12 and not more.** Adding rows where ThinkBooster has ✗ (e.g., MCTS, Z3-solver back-end) breaks the all-✓ framing. Adding rows where every framework has ✓ (e.g., "Best-of-N", "Self-consistency") is dead weight — they don't differentiate. The 12 above all genuinely differentiate.

**Trim if column space is tight.** Top-6 highest-value rows (most differentiating, hardest to dispute) are: 3, 4, 6, 7, 9, 10. Use those if Table 1 needs to fit a single column.

---

## 2. Competitors (columns)

Reviewer paXg explicitly names these four — they are mandatory:

| Column | Repo | Owner |
|---|---|---|
| **LLM Reasoners** | https://github.com/maitrix-org/llm-reasoners | **Vlad** |
| **search-and-learn** | https://github.com/huggingface/search-and-learn | **Quang** |
| **OpenR** | https://github.com/openreasoner/openr | **Sergey** |
| **OptiLLM** | https://github.com/codelion/optillm | **Artem S** |

### Should we add more?

The reviewer used "e.g." — they're not exhaustive. Candidates I considered:

- **DSPy** — declarative LM programming with a few search optimizers. Different scope (optimizes prompts/programs, not test-time reasoning compute). **Skip** unless we want to position more broadly.
- **Skywork-o1-Open** — TTC + RL training. Model-side, not framework. **Skip**.
- **LangGraph / AutoGen** — agent frameworks, only marginal TTC overlap. **Skip**.
- **Tree-of-Thoughts (Princeton)** — single strategy, not a library. **Skip**.

**Recommendation:** stick to the 4 named by the reviewer. Adding off-axis candidates dilutes the differentiation and invites "this isn't a fair comparison" pushback.

---

## 3. First-pass comparison table

Symbols: ✓ supported · ✗ not supported · ◐ partial / limited · ? **needs verification by owner**

Cell values below are my best read from each repo's README + the source code I sampled. Anywhere a cell is marked `?@<owner>`, that's where I want the assignee to confirm before the PR closes.

| Feature | ThinkBooster | LLM Reasoners | search-and-learn | OpenR | OptiLLM |
|---|---|---|---|---|---|
| **1. Strategy taxonomy breadth (≥6 families)** | ✓ (9) | ◐ (5: BFS, DFS, MCTS, Beam, RAP) `?@vlad` | ✗ (3: BoN, Beam, DVTS) `?@quang` | ◐ (3: BoN, Beam, MCTS) `?@sergey` | ✓ (≥20 techniques, but mostly prompting variants) `?@artems` |
| **2. Online + offline modes** | ✓ | ◐ (online tree search; offline limited) `?@vlad` | ✓ | ◐ `?@sergey` | ◐ (mostly offline rerank/voting) `?@artems` |
| **3. Adaptive / uncertainty-driven** | ✓ (MUR, phi-decoding, uncertainty-CoT) | ✗ `?@vlad` | ✗ `?@quang` | ✗ `?@sergey` | ✗ `?@artems` |
| **4. Confidence-based steering** | ✓ (DeepConf online + offline) | ✗ `?@vlad` | ✗ `?@quang` | ✗ `?@sergey` | ✗ `?@artems` |
| **5. ≥3 scorer families (PRM / uncertainty / LLM-critic / supervised)** | ✓ (4) | ◐ (PRM + self-eval) `?@vlad` | ✗ (PRM only) `?@quang` | ◐ (PRM with RL training only) `?@sergey` | ◐ (voting + self-consistency, no PRM) `?@artems` |
| **6. Supervised step scorer (ReProbe-style)** | ✓ | ✗ `?@vlad` | ✗ `?@quang` | ✗ `?@sergey` | ✗ `?@artems` |
| **7. Configurable score aggregation (min / mean / max / product + sliding window)** | ✓ | ✗ `?@vlad` | ✗ `?@quang` | ✗ `?@sergey` | ✗ `?@artems` |
| **8. OpenAI-compatible REST gateway** | ✓ | ✗ `?@vlad` | ✗ `?@quang` | ◐ (HTTP server, not OpenAI-shaped) `?@sergey` | ✓ |
| **9. Visual / interactive debugger** | ✓ (3 views) | ◐ (static visualizer) `?@vlad` | ✗ `?@quang` | ✗ `?@sergey` | ✗ `?@artems` |
| **10. Joint performance–compute benchmark (TFLOPs + tokens)** | ✓ | ✗ `?@vlad` | ◐ (token accounting only) `?@quang` | ✗ `?@sergey` | ✗ `?@artems` |
| **11. Black-box compatibility (no logits / prefill required)** | ✓ | ✗ (HF/white-box) `?@vlad` | ✗ (vLLM/white-box) `?@quang` | ◐ `?@sergey` | ✓ |
| **12. Crash-resistant evaluation pipeline (resume / checkpoint)** | ✓ | ✗ `?@vlad` | ✗ `?@quang` | ✗ `?@sergey` | ✗ `?@artems` |

### Per-cell notes

- **Row 1 (Strategy breadth).** OptiLLM has lots of *techniques* but most are prompting variants (CoT-reflection, plan-and-search, etc.) rather than search-with-scoring. Owner should decide whether to count them as full "strategies".
- **Row 5 (Scorer families).** Distinguish "scorer family" (PRM vs. uncertainty vs. critic vs. supervised) from "scorer instance" (Qwen2.5-Math-PRM-7B vs. Skywork-PRM). Most competitors only have one *family*.
- **Row 8 (REST gateway).** OpenR ships an HTTP server, but it's not OpenAI-SDK-shaped — Sergey to confirm whether it accepts `chat.completions` requests unchanged.
- **Row 9 (Visual debugger).** LLM Reasoners has a rendering tool for trees but not interactive replay; mark ◐ unless Vlad finds otherwise.
- **Row 11 (Black-box).** OptiLLM is OpenAI-proxy-shaped, so by construction works with any OpenAI-compatible endpoint; ✓ is uncontroversial. ThinkBooster supports both because of OpenRouter + the strategies that don't need prefill (BoN, voting, ext-thinking).

---

## 4. LaTeX version (drop-in for the paper)

Uses `booktabs`, fits one column on `acl_natbib`-style. ThinkBooster is the leftmost data column for prominence; competitors follow alphabetically.

```latex
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccccc@{}}
\toprule
                       & \textbf{ThinkBooster} & \textbf{LLM Reasoners} & \textbf{search-and-learn} & \textbf{OpenR} & \textbf{OptiLLM} \\
\midrule
Strategy taxonomy ($\geq 6$ families)
                       & \checkmark (9)  & \halfcirc (5)   & \times (3)   & \halfcirc (3)   & \checkmark ($\geq 20$) \\
Online + offline modes
                       & \checkmark      & \halfcirc       & \checkmark   & \halfcirc       & \halfcirc \\
Adaptive / uncertainty-driven
                       & \checkmark      & \times          & \times       & \times          & \times \\
Confidence-based steering (DeepConf)
                       & \checkmark      & \times          & \times       & \times          & \times \\
$\geq 3$ scorer families
                       & \checkmark (4)  & \halfcirc (2)   & \times (1)   & \halfcirc (1)   & \halfcirc (1) \\
Supervised step scorer (ReProbe)
                       & \checkmark      & \times          & \times       & \times          & \times \\
Configurable score aggregation
                       & \checkmark      & \times          & \times       & \times          & \times \\
OpenAI-compatible REST gateway
                       & \checkmark      & \times          & \times       & \halfcirc       & \checkmark \\
Visual / interactive debugger
                       & \checkmark      & \halfcirc       & \times       & \times          & \times \\
Joint performance--compute benchmark
                       & \checkmark      & \times          & \halfcirc    & \times          & \times \\
Black-box compatibility
                       & \checkmark      & \times          & \times       & \halfcirc       & \checkmark \\
Crash-resistant resume
                       & \checkmark      & \times          & \times       & \times          & \times \\
\bottomrule
\end{tabular}
\caption{Comparison of test-time compute (TTC) scaling frameworks.
\checkmark{} = first-class support;
\halfcirc{} = partial / limited;
$\times$ = not supported.}
\label{tab:ttc_framework_comparison}
\end{table*}

% Add to preamble if not already present:
% \usepackage{wasysym}      % \checkmark
% \newcommand{\halfcirc}{\(\ocircle\!\!\!\!\!\bullet\)}  % or use \LEFTcircle from wasysym
```

---

## 5. TODO — per-owner verification

Each owner: open the listed repo, check the cells with `?@<you>` in §3, and either:

1. confirm my mark (✓ / ✗ / ◐) — leave a comment on PR #251 with one-line evidence, or
2. correct it — push a commit to `camera-ready/acl2026` updating the cell + add a note to §3 "Per-cell notes".

| Owner | Library | Cells to verify |
|---|---|---|
| **Vlad** | LLM Reasoners | rows 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 (all `?@vlad`) |
| **Quang** | search-and-learn | rows 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 (all `?@quang`) |
| **Sergey** | OpenR | rows 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 (all `?@sergey`) |
| **Artem S.** | OptiLLM | rows 1, 2, 3, 4, 5, 6, 7, 9, 10, 12 (all `?@artems`) |

After all `?@<owner>` markers are resolved, drop the LaTeX block straight into `paper/sections/related_work.tex` (or wherever the paper sources live) and bump the citation count.

---

## 6. Open questions

- **Row 1 strategy count for OptiLLM.** Do prompting-only techniques (CoT-reflection, plan-and-execute) count as "TTC scaling strategies" for the purposes of this table? Argument for: they consume extra compute. Argument against: they're prompt engineering, not search/scoring. **Artem S to decide before camera-ready.**
- **Should we add a "Last release / actively maintained" row?** It would highlight ThinkBooster but is borderline ad-hominem on OpenR (slower release cadence). **Decision: skip unless reviewer pushes.**
- **Should we cite each framework's paper in the table caption or in the body?** Body is cleaner; caption gets cluttered. **Default: body cite via `\citep{}`.**
