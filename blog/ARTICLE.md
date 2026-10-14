# ThinkBooster: unlock your LLM's Pro reasoning mode by changing one URL

*Plus a benchmark that finally tells you what the extra accuracy costs.*

*Read the paper on [arXiv](https://arxiv.org/abs/2606.06915).*

> _[Figure: hero — the endpoint gateway diagram. Upload `blog/endpoint_hero.png` (1392×889, 149 KB; exported from `images/endpoint.pdf`).]_

---

Every frontier model now ships a "thinking" mode. Ask o1, R1, or Qwen3 a hard question and it spends more compute before answering, and usually does better for it. Test-time compute scaling is the name for that move: you pay more at answer time, not training time, to get a better answer.

The catch is that "spend more compute" is not one method. You can sample ten answers and keep the best. You can search a tree of reasoning steps and prune the weak branches. You can let the model run until it is confident, then stop. You can add compute only on the steps where it hesitates. These cost very different amounts, and most papers don't report the cost. So the question every practitioner actually has, *for the accuracy I need, which method is cheapest?*, has no clean answer.

ThinkBooster is built to answer it.

## What ThinkBooster is

ThinkBooster is an open-source (MIT) framework for test-time compute scaling. It puts nine scaling strategies and four scorer families behind one API, measures accuracy and compute together, and ships as an OpenAI-compatible proxy with a visual debugger.

The one design idea worth knowing: strategy and scorer are separate. The strategy decides the shape of the search, such as how many samples to draw, whether to follow a single line or branch into a tree, and when to stop. The scorer decides which partial answer looks best. ThinkBooster ships four kinds of scorer:

- a **process reward model**: a separate critic model trained to grade reasoning steps;
- the model's own **confidence**, read straight off its token probabilities, with no extra model to run;
- an **LLM-as-judge**: ask a model to grade the work;
- **ReProbe**: a light supervised probe on the model's internal states (our own work; more on it below).

Any strategy pairs with any scorer through a registry, so you can ask for "beam search with a reward model" or "best-of-N with confidence" without writing glue code. That registry is also the extension point: add a strategy or scorer once as a subclass, and the gateway, the benchmark, and the debugger all pick it up. Generation runs on vLLM underneath, so the same code goes from one GPU to multi-node.

> _[Figure: the nine strategies (`blog/fig_strategies.png`)]_

## The hook, in one line of code

The proxy is the part most people touch first. ThinkBooster sits in front of your model as an OpenAI-compatible endpoint, and the strategy and scorer live in the URL:

```
base_url = "<THINKBOOSTER_ENDPOINT>/v1/beam_search/prm"
```

Point your existing OpenAI client at that, and the same model now runs beam search scored by a reward model. Change it to `.../v1/offline_bon/entropy` and you have swapped the entire scaling method, with no other edit to your code. Agents, copilots, and enterprise stacks already speak OpenAI, so you add reasoning scaling by editing a string, and you keep the compute budget in your hands.

## Does it actually work?

Three results from the benchmark are worth pulling out, because each one cuts against the obvious intuition.

**Confidence can beat a trained reward model, and confidence is free.** On HumanEval+, pairing the MUR strategy (which spends extra compute only on uncertain steps) with a plain entropy signal takes Qwen3-8B from 79.3 to 88.8, the best coding score in the study and higher than any reward-model setup. Entropy is read off the tokens the model already produced, so it adds almost nothing, while a reward model is a second network you have to run. Against the most accurate reward-model setup, beam search with a PRM, confidence is both more accurate and under a quarter of the compute. The scope is narrow: this is a coding result. On math, the cheap signal does not win; there the strongest results come from a PRM, or from self-consistency on the hardest sets.

> _[Figure: Qwen3-8B on HumanEval+ — accuracy vs. compute (`blog/fig_qwen3_humaneval.png`)]_

**Spending more does not reliably buy more.** With the cheap scorers, beam search trails plain best-of-N and even self-consistency, despite searching much harder. Its one strong configuration, beam search with a reward model, reaches the top math accuracy on several of the benchmarks, such as OlympiadBench and Gaokao, though best-of-N with the same reward model ties or beats it on the hardest sets. And it costs 17 to 24 times more than best-of-N to get there, often for a tie or a single point. For most budgets, best-of-N with a reward model sits at the better accuracy-per-FLOP point.

> _[Figure: Qwen2.5-Math — accuracy vs. compute, aggregate over MATH-500, OlympiadBench, Gaokao (`blog/fig_qwen25_aggregate.png`)]_

**A drop-in change can improve real engineering output.** On GPT-OSS-120B writing CUDA kernels, adding best-of-N with a reward model raised end-to-end correctness from 26 to 30 and cut syntax errors by five points. Compilation rate dipped by one point, consistent with the reward model favoring more ambitious kernels that are likelier to fail to compile, a trade the correctness gain pays for.

> _[Figure: GPT-OSS-120B results (`blog/fig_gpt_oss.png`)]_

None of this shows up in an accuracy column on its own. Because ThinkBooster's benchmark logs TFLOPs and tokens next to accuracy, you can see the trade and pick the point you can afford.

## ReProbe, on a model the paper never tested

The internal-state scorer above, ReProbe, is its own line of work: *ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models*. Instead of a full reward model or a single confidence number, it trains a small probe to read a model's hidden states and predict whether a reasoning step is on track. No other framework in our comparison offers it as a first-class scorer, and because ThinkBooster keeps the model backend swappable, we can point it at models the original paper never benchmarked.

We did that with K2-Think-V2, the reasoning model from MBZUAI and G42. With no changes to the model, ReProbe-guided best-of-N inside ThinkBooster lifts K2-Think-V2's MBPP+ pass rate over plain single-shot decoding, on both the base tests and the harder plus set.

> _[Figure: K2-Think-V2 + ReProbe best-of-N vs single-shot on MBPP+ base (`blog/fig_k2_base.png`)]_

> _[Figure: K2-Think-V2 + ReProbe best-of-N vs single-shot on MBPP+ plus (`blog/fig_k2_plus.png`)]_

The point it makes is the one that matters here: a new model and a research-grade scorer drop into the same framework and the same benchmark with no special-casing.

## Which one should you use?

A rough rule of thumb from the numbers above:

- **Closed API, no access to logits:** best-of-N, majority voting, or extended thinking.
- **Self-hosted and cost-sensitive:** best-of-N or MUR with a confidence scorer, which is free to compute and needs no second model.
- **Want top accuracy on math and can pay for it:** beam search with a reward model, though best-of-N with the same model is often as good for far less.

## How it compares to OptiLLM and the rest

ThinkBooster is not the only tool here, and the closest one is good. OptiLLM is an OpenAI-compatible proxy with a wide menu of inference techniques, and by a loose count it covers more tricks than ThinkBooster does. Our framework table uses a strict rule, counting genuine search, scoring, and decoding methods and excluding prompt-engineering scaffolds, and by that rule it is nine strategies to OptiLLM's seven. ThinkBooster's nine families span more than twenty recent methods, and OptiLLM likewise bundles many prompt and inference techniques. We are not claiming more tricks.

What no other framework offers all at once:

- uncertainty and ReProbe-style internal-state scoring as first-class, swappable scorer families, built on LM-Polygraph, our group's uncertainty library;
- first-class process-reward-model scoring and FLOPs-level compute accounting, which OptiLLM does not foreground;
- a benchmark that reports accuracy and compute together, with bundled, judged datasets across math, coding, and science;
- a visual debugger for reasoning trajectories.

The other frameworks are narrower. LLM Reasoners is a strong modular research library, but it has no OpenAI endpoint, no native vLLM backend, and no joint performance–compute benchmark. The rest are narrower in scope, each oriented around a specific algorithm, a reproduction recipe, or visualization. The cells in our comparison are our own assessment rather than an audited score, but the gaps in uncertainty scoring and compute accounting are real.

> _[Figure: framework comparison (`blog/fig_ttc_frameworks.png`)]_

## See why a trajectory wins: the visual debugger

A scaling run makes decisions you never normally see: which candidates it sampled, what each one scored, which branch it kept, where it stopped. ThinkBooster ships an interactive debugger that opens all of it in the browser, and it is live: open the [demo](http://demo-thinkbooster.nlpresearch.group), pick a provider, model, strategy, and scorer, type in a problem, and run it. The run comes back as a reasoning timeline: every step with its score, the candidates the strategy considered, and which one was selected versus pruned. Click a step and the inspector shows the full candidate text and the number behind the decision.

> _[Figure: the debugger — run config, strategy comparison, reasoning timeline, and step inspector in one place (`blog/fig_debugger_a.png`)]_

The trajectory tree shows the whole search at once. The highlighted path is the trajectory that was selected; the other branches were explored and dropped. It is the quickest way to see how beam search or best-of-N spends its compute and where it walks away from a dead end.

> _[Figure: the trajectory tree — selected path in orange, pruned branches in gray (`blog/fig_debugger_b.png`)]_

You can also run two strategies on the same problem under the same compute budget and compare them side by side, which turns "which method is better" from an argument into something you can watch.

## Try it in five minutes

Install:

```bash
pip install thinkbooster
```

**Option 1 — run it as a service and call the endpoint.** Stand the service up locally (the [local service guide](https://github.com/IINemo/thinkbooster/blob/main/docs/service/running_locally.md) covers building it and calling it as an endpoint), then point any OpenAI client at it. The strategy and scorer live in the URL, so the rest of your code does not change:

```python
from openai import OpenAI

client = OpenAI(
    base_url="<THINKBOOSTER_ENDPOINT>/v1/beam_search/prm",
    api_key="<YOUR_API_KEY>",
)
response = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B",
    messages=[{"role": "user", "content":
        "Find the number of ordered pairs (x, y) of "
        "positive integers satisfying x + 2y = 2xy."}],
    extra_body={"max_tokens": 8192, "tts_beam_size": 4},
)
print(response.choices[0].message.content)
```

**Option 2 — import the pieces as a library.** Compose a model, a step generator, and a strategy yourself, then run a question through it:

```python
import os

from lm_polygraph.utils.generation_parameters import GenerationParameters
from thinkbooster.models.blackboxmodel_with_streaming import BlackboxModelWithStreaming
from thinkbooster.generators.api import StepCandidateGeneratorThroughAPI
from thinkbooster.strategies.strategy_self_consistency import StrategySelfConsistency

# 1. A model behind any OpenAI-compatible endpoint (OpenRouter here)
model = BlackboxModelWithStreaming(
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    model_path="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    supports_logprobs=True,
    generation_parameters=GenerationParameters(
        max_new_tokens=2048, temperature=0.7, top_p=0.8,
    ),
)

# 2. A step generator over that model
generator = StepCandidateGeneratorThroughAPI(
    model=model, max_new_tokens=2048, temperature=0.7, top_p=0.8,
)

# 3. A strategy: self-consistency over 8 sampled paths
strategy = StrategySelfConsistency(
    step_generator=generator, num_paths=8, data_name="math",
)

# 4. Run it on a question
request = [
    {"role": "system", "content":
        "Please reason step by step, and put your final answer within \\boxed{}."},
    {"role": "user", "content": "What is 2^10 + 3^5?"},
]
results = strategy.generate_trajectories_batch(
    requests=[request], sample_indices=[0],
)
print(results[0]["trajectory"])
print("Answer:", results[0]["extracted_answer"])
```

From there you can run the compute-aware benchmark with one command to get accuracy and TFLOPs side by side, or open the visual debugger from the section above at `localhost:8001/debugger`.

## Limitations

The dynamic, confidence-driven strategies (MUR, DeepConf, uncertainty CoT) need logits or hidden states, so they want open-weight or self-hosted models. Against a closed API you get the black-box subset: best-of-N, majority voting, extended thinking, and LLM-as-judge scoring. Splitting native, unstructured "thinking" traces into clean steps is still hard, which can affect step-level scoring. And the evidence so far covers math, coding, and science QA; compute is reported as theoretical TFLOPs and tokens, with wall-clock logged but not yet studied.

## Why you should try it

Test-time compute scaling is here to stay; the useful question is which method, at what cost. ThinkBooster lets you measure that and ship the answer by changing a URL. If you research reasoning, you get a reproducible, compute-aware benchmark and scorers, including uncertainty and ReProbe, that you will not find elsewhere. If you build with LLMs, you get a drop-in proxy with the budget in your control.

You can try it in your browser right now, no install needed: [the live demo](http://demo-thinkbooster.nlpresearch.group), or watch the [screencast](https://www.youtube.com/watch?v=7Idrcs_4kfQ) first. To run it locally:

```bash
pip install thinkbooster
```

Code and docs are on [GitHub](https://github.com/IINemo/thinkbooster); the paper is on [arXiv](https://arxiv.org/abs/2606.06915).

