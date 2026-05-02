# ACL 2026 — ThinkBooster (Submission #191) — Acceptance & Reviews

---

## Paper Decision

> Decision by Program Chairs · 24 Apr 2026, 16:48 (modified: 24 Apr 2026, 22:14)
> Visible to: Program Chairs, Authors

**Decision:** Accept

---

## Meta Review of Submission 191 by Area Chair `9gJd`

> Meta Review by Area Chair 9gJd · 12 Apr 2026, 07:07 (modified: 24 Apr 2026, 22:14)
> Visible to: Senior Area Chairs, Area Chairs, Authors, Program Chairs

### Metareview

ThinkBooster is a unified framework for test-time compute (TTC) scaling of LLM reasoning, combining a modular Python library of strategies and scorers, a joint performance-compute benchmark, and an OpenAI-compatible endpoint gateway with a visual debugger for reasoning trajectory inspection.

**Pros:**

- Addresses a real and practical need, bringing multiple TTC scaling strategies under the same unified framework
- Solid engineering choices with OpenAI-compatible gateway that could enable adoption and a modular design with clean open-source artifacts

**Cons:**

- No comparison to existing TTC frameworks (LLM Reasoners, search-and-learn, OpenR, OptiLLM) would help make the contribution of this paper stronger.
- Agreeing with Reviewer 9UwS, critique-based scaling methods (e.g. self-correction, self-verification) are a notable missing family from the strategy taxonomy

**Recommendations for camera ready:** For the camera ready, the authors should: add wall-clock latency analysis alongside the TFLOPs/token efficiency metrics; ensure the live demo is accessible; discuss the white-box access requirements as a limitation for fully hosted models; and acknowledge the current scope limitation to math/coding/science tasks.

---

## A Unified Python Toolkit for Test-Time Scaling of LLM Reasoning

> Official Review by Reviewer paXg · 04 Apr 2026, 20:32 (modified: 24 Apr 2026, 22:14)
> Visible to: Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer paXg, Authors

### Summary

This paper proposes ThinkBooster, a unified framework for test-time compute (TTC) scaling of LLM reasoning. It addresses two key gaps in current TTC scaling research: (1) the lack of standardized evaluation that jointly considers both performance and compute efficiency, and (2) the absence of a unified implementation across diverse TTC scaling methods. ThinkBooster provides a modular Python library implementing common TTC scaling strategies and scorers, along with a benchmark that evaluates both task performance and compute efficiency (in TFLOPs and token cost).

### Review

**Overall Evaluation**

ThinkBooster presents a well-engineered and practically useful unified framework for test-time compute scaling of LLM reasoning. However, the paper lacks sufficient comparison with existing TTC scaling frameworks and the evaluation is limited to open-source models without wall-clock latency analysis, and the live demo is also inaccessible during review.

**Pros**

Good system design and practical engineering contribution. ThinkBooster cleanly separates different TTC modules and supports both swift implementation with OpenAI compatible SDK, making it practical for both research and production use.

Performance–compute efficiency benchmarking. The framework provides a comprehensive and systematic evaluation of existing TTC scaling methods, jointly measuring task performance and compute efficiency.

Clean open-source artifacts. The demo video and code repository appear well-organized and of good quality, which lowers the barrier for adoption and implementation.

**Cons**

Insufficient comparison with existing frameworks. Several TTC scaling frameworks already exist e.g., LLM Reasoners (MCTS, BFS/DFS, beam search with multiple backends), HuggingFace's search-and-learn (Best-of-N, beam search, DVTS), OpenR (Best-of-N, beam search, MCTS with PRM), and OptiLLM (an OpenAI-compatible proxy with 20+ techniques). The paper lacks a systematic comparison with these alternatives in terms of strategy coverage, scorer diversity, and deployment design, making it difficult to assess ThinkBooster's unique positioning.

Only open-source models evaluated. All tested models (Qwen2.5-Math-7B, Qwen3-8B, GPT-OSS-120B) are open-source. Commercial models, which are common in production settings, are not evaluated.

Efficiency measured only in tokens/TFLOPs, lacking wall-clock time. While reporting token counts and TFLOPs is reasonable, it is difficult to assess practical feasibility from these numbers alone. R.g., a 10x or 100x compute ratio does not tell practitioners whether the actual latency is acceptable. Supplementing with wall-clock execution time under specific hardware configurations would make the efficiency analysis more informative.

Demo link is inaccessible. The demo URL provided in the footnotes (`http://demo-thinkbooster.nlpresearch.group`) is currently not working. For a demo track submission, being unable to access the live demo during review is a notable issue.

### Reasons To Accept

ThinkBooster offers a well-designed, modular framework that unifies fragmented TTC scaling implementations behind a consistent API, filling a practical gap for both researchers and practitioners. The OpenAI-compatible endpoint gateway enables drop-in adoption with minimal friction. The joint performance–compute benchmark sets a good precedent for the field by systematically evaluating efficiency alongside accuracy.

**Rating:** 6: Marginally above acceptance threshold

### Reasons To Reject

The paper lacks comparison with several existing TTC scaling frameworks (LLM Reasoners, search-and-learn, OpenR, OptiLLM), making it difficult to assess its unique contributions. The evaluation is limited to open-source models with no commercial model experiments, and efficiency is reported only in theoretical TFLOPs without wall-clock latency. The live demo is inaccessible during review, which is a significant concern for a demo track submission.

### Questions And Additional Feedback

**Questions**

1. **Comparison with existing frameworks.** Could the authors provide a more detailed comparison to better position ThinkBooster's unique contributions against these alternatives?
2. **Wall-clock latency measurement.** The current efficiency analysis relies on theoretical TFLOPs and token counts. Could the authors supplement this with wall-clock execution time under specific hardware configurations?
3. **Commercial model evaluation.** All experiments use open-source models. Given that the framework is designed as an OpenAI-compatible proxy, have the authors considered evaluating commercial models (e.g., GPT-5, Claude)?
4. **PRM interchangeability across tasks.** Does the framework support easy swapping of PRMs for different task domains? The current experiments use a math-trained PRM (Qwen2.5-Math-PRM-7B) even for coding tasks. For harder or domain-specific tasks, a stronger or domain-matched PRM may be needed — how straightforward is this replacement within the framework?

### Form fields

- **Needs Ethical Review:** No
- **Reproducibility:** 4 — They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.
- **Software Or Live Demo:** 4 — Useful: I would recommend the new software / live demo to other researchers or developers for their ongoing work.
- **Datasets:** 3 — Potentially useful: Someone might find the new datasets useful for their work.
- **Overall Assessment:** 6: Marginally above acceptance threshold

---

## Review

> Official Review by Reviewer yXVQ · 01 Apr 2026, 05:26 (modified: 24 Apr 2026, 22:14)
> Visible to: Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer yXVQ, Authors

### Summary

This paper presents THINKBOOSTER, a unified framework for test-time compute (TTC) scaling of LLM reasoning. It addresses the problem that existing TTC scaling strategies—like best-of-N, tree-of-thought, and self-consistency—are fragmented, evaluated inconsistently, and rarely consider compute-efficiency trade-offs. THINKBOOSTER provides a modular Python library with state-of-the-art scaling strategies and scorers, a benchmark for joint performance–compute evaluation, an OpenAI-compatible endpoint gateway for real-world deployment, and a visual debugger for inspecting reasoning trajectories. Its main contributions are enabling seamless test-time scaling, principled evaluation of performance–compute trade-offs, and practical integration into applications like coding assistants and mathematical problem solvers, demonstrating improved accuracy and efficiency on math, scientific, and programming benchmarks.

### Review

**Review of "THINKBOOSTER: A Unified Framework for Test‑Time Compute Scaling of LLM Reasoning"**

**Summary**

The paper introduces THINKBOOSTER, a modular library and evaluation framework for test-time compute (TTC) scaling of large‑language‑model reasoning. It addresses limitations in existing TTC strategies (e.g., best‑of‑N, self‑consistency, tree‑of‑thought) by unifying them under a common API, providing principled performance–compute benchmarking, real‑world endpoints, and tools for debugging reasoning trajectories. The work demonstrates improved accuracy and more efficient compute use on math, science, and coding benchmarks.

**Evaluation**

*Quality.* The implementation appears engineered with practical utility in mind, offering a standard interface to plug in diverse scaling strategies and scorers. The benchmarking setup considers both reward and computational cost, which is important for realistic evaluation. However, the quality of empirical validation depends on the breadth and depth of benchmarks chosen (details not in summary). The paper defines relevant metrics and trade‑offs, e.g.:

> *(equation rendered in original review — not preserved in plaintext copy)*

which formalizes how added compute translates into performance improvement.

*Clarity.* The presentation is generally clear, with modular descriptions of components (strategy, scorer, gateway, debugger). Architectural diagrams and code examples (if included) likely help clarify usage. One area that may benefit from more explicit description is how scalability and cost are measured consistently across different strategies.

*Originality.* The work's main novelty lies in systematizing disparate TTC strategies into a unified framework with common interfaces and joint performance–compute evaluation. While existing work explores many scaling strategies in isolation, THINKBOOSTER's contribution is in engineering unification and tooling, rather than new algorithmic scaling methods per se.

*Significance.* The framework has practical significance for practitioners and researchers needing to compare and deploy TTC scaling methods. By exposing a shared API and benchmarking tools, THINKBOOSTER can foster more comparable research and real‑world adoption of compute‑aware reasoning strategies. Its integration with production endpoints and a visual debugger also aids interpretability and deployment.

**Pros**

- Unified API for diverse TTC strategies (best‑of‑N, self‑consistency, tree‑of‑thought).
- Principled performance–compute evaluation, enabling trade‑off analysis rather than raw accuracy.
- Modular design encouraging extensibility (swap strategy, scorer, or model backend).
- Includes production integration (OpenAI‑compatible endpoint gateway) and visual debugging of reasoning trajectories.
- Demonstrated practical gains on reasoning benchmarks in math, science, and programming domains.

**Cons**

- Focus is more on systems engineering and benchmarking rather than proposing fundamentally new TTC algorithms.
- Empirical evaluation quality depends heavily on selected benchmarks and cost models; generalization to other tasks may vary.
- The cost model may not account for all real‑world constraints (e.g., memory, latency, parallelization).
- Comparisons with the latest adaptive or learned scaling strategies beyond classic ones may be limited.
- Users must still choose scoring functions and compute trade‑offs, which may require domain expertise.

**Overall**

THINKBOOSTER presents a practical and well‑engineered framework that consolidates test‑time compute scaling approaches under a coherent API and evaluation paradigm. It is clear, original in its system perspective, and significant for reproducible and comparable reasoning research. The work's strengths lie in its tooling and benchmarking infrastructure, which can support both research and deployment. However, its contribution is largely in unification and tooling rather than novel algorithms.

**Recommendation: Accept** — the contribution is valuable for the NLP community's shift toward compute‑aware, benchmarkable reasoning.

### Reasons To Accept

The paper's strengths are its unified, modular framework for test-time compute (TTC) scaling, standardized benchmarking of performance–compute trade-offs, and practical deployment through an OpenAI-compatible endpoint and visual debugger. Presenting it benefits the NLP community by providing a reproducible, comparable, and developer-friendly system for evaluating and deploying adaptive LLM reasoning strategies, enabling more systematic research and practical adoption of compute-aware LLM reasoning methods.

**Rating:** 6: Marginally above acceptance threshold

### Reasons To Reject

Weaknesses of the paper include its focus on systems integration and benchmarking rather than novel TTC algorithms, reliance on specific LLMs and benchmark datasets (math, science, coding), and white-box strategy requirements that may not generalize to all hosted LLMs. Presenting it may risk overstating generalizability and real-world impact, as performance–compute trade-offs could vary for other models, domains, or deployment environments.

### Questions And Additional Feedback

1. How well does THINKBOOSTER generalize to LLMs outside math, coding, and scientific QA, such as long-context or tool-using models?
2. Are there plans to support fully black-box LLMs without white-box signals like logits or prefill options?
3. How does the visual debugger scale for very long reasoning trajectories or multiple simultaneous requests?
4. Can the framework handle dynamic or adaptive compute budgets in real-time applications with latency constraints?
5. How would THINKBOOSTER integrate with multi-agent or chain-of-tool pipelines, and are there plans for such evaluations?

### Form fields

- **Needs Ethical Review:** Yes
- **Reproducibility:** 3 — They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.
- **Software Or Live Demo:** 3 — Potentially useful: Someone might find the new software / live demo useful for their work.
- **Datasets:** 3 — Potentially useful: Someone might find the new datasets useful for their work.
- **Overall Assessment:** 6: Marginally above acceptance threshold

---

## Official Review of ThinkBooster

> Official Review by Reviewer 9UwS · 30 Mar 2026, 06:36 (modified: 24 Apr 2026, 22:14)
> Visible to: Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer 9UwS, Authors

### Summary

This paper presents a modular and practical framework for improving LLM reasoning by allocating more computation at inference time. The paper tackles the current fragmentation in test-time scaling methods by offering a unified system that supports multiple reasoning strategies, scoring mechanisms, and evaluation tools.

### Review

See "Reasons To Accept" and "Reasons To Reject"

### Reasons To Accept

Practical and easy to integrate: The OpenAI-compatible gateway is a strong design choice because it makes the framework easy to adopt in existing applications without major engineering changes.

Well-rounded framework (comprehensive for TTS methods): The paper does a good job of combining methods, deployment, and benchmarking in one system, which makes it more complete and useful than a narrowly scoped demo.

Strong emphasis on transparency: The visual debugger is especially appealing, as it gives users insight into reasoning trajectories and makes the system more interpretable and easier to analyze.

**Rating:** 6: Marginally above acceptance threshold

### Reasons To Reject

Evaluation scope seems somewhat narrow: The current focus on domains like math and coding is promising, but it would be even stronger to see broader validation on other tasks.

Another line of scaling is "critique" based methods [1, 2]. It is better to cover this kind of TTS as well.

[1] Training Language Models to Self-Correct via Reinforcement Learning. ICLR 2025

[2] Trust, But Verify: A Self-Verification Approach to Reinforcement Learning with Verifiable Rewards. NeurIPS 2025

### Questions And Additional Feedback

No

### Form fields

- **Needs Ethical Review:** No
- **Reproducibility:** 4 — They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.
- **Software Or Live Demo:** 4 — Useful: I would recommend the new software / live demo to other researchers or developers for their ongoing work.
- **Datasets:** 1 — No usable datasets submitted.
- **Overall Assessment:** 6: Marginally above acceptance threshold

---

## Camera-ready punch list (synthesised from all three reviewers + meta)

This is an editorial summary derived from the verbatim reviews above; not part of the original OpenReview content.

1. **Wall-clock latency** alongside TFLOPs/token (paXg, meta).
2. **Live demo accessibility** — fix `http://demo-thinkbooster.nlpresearch.group` (paXg, meta).
3. **Comparison with existing TTC frameworks** — LLM Reasoners, search-and-learn, OpenR, OptiLLM (paXg, meta).
4. **Critique-based scaling family** — self-correction, self-verification; refs [Kumar et al. ICLR 2025], [Trust-But-Verify NeurIPS 2025] (9UwS, meta).
5. **Limitations:** white-box access requirements (meta), task scope math/coding/science (meta, yXVQ).
6. **(Optional) Commercial models** — GPT-5, Claude via OpenAI-compatible proxy (paXg).
7. **(Optional) PRM interchangeability** — clarify swap story for non-math domains (paXg).
8. **(Optional) Reproducibility** — tighten parameter specifications flagged by yXVQ.
9. **(Optional) Generalisation** — long-context / tool-using / black-box models, multi-agent pipelines (yXVQ).
