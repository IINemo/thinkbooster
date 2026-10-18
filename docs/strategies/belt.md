# BELT: belief scoring and early stopping

BELT learns a reusable representation of a reasoning prefix. Its four readouts
describe correctness now (`q`), correctness at natural completion (`v`), the
possibility of a later correct answer (`r`), and answer stability (`stability`).
A separately fitted linear policy uses either the hidden representation (H) or
the belief logits (B) to decide when to stop thinking.

[BELT paper and reproduction repository](https://github.com/EdisonNi-hku/BELT)
· [Recommended Qwen checkpoints](https://huggingface.co/JingweiNi/BELT-Qwen3-1.7B-ReProbeInit)
· [Checkpoint inventory and hashes](https://github.com/EdisonNi-hku/BELT/blob/main/docs/CHECKPOINTS.md)

This adapter supports **Qwen3-1.7B**, both published initialization families,
and all three training seeds. It provides a Python scorer, an early-stopping
strategy and Hydra evaluation configuration. OLMo and the service gateway are
outside this adapter's current backend support.

## Installation

Install ThinkBooster and its adapter dependencies:

```bash
pip install -e ".[belt]"
```

The reader runs through BELT's original implementation. Keep its source
checkout installed in editable mode so that the native reader files remain
available:

```bash
git clone https://github.com/EdisonNi-hku/BELT.git ../BELT
git -C ../BELT checkout fc9b0670e430ed3519eb0df25aef0f5788a8b29c
pip install --no-deps -e ../BELT
```

The adapter temporarily loads that checkout's native `luh` and `train_luh`
packages during initialization, then restores the previous modules and
`sys.path`, including when loading fails. A subsequent `import luh` therefore
uses the application's original reader. BELT initializers are serialized;
do not import `luh` or `train_luh` in another thread during initialization.
Native scoring retains its own module references and needs no further imports
of those packages. The separate BELT runtime and model assets retain their
original notices and terms; see [BELT's source notice](https://github.com/EdisonNi-hku/BELT/blob/main/NOTICE.md).

The pinned source and probe packages currently live in personal GitHub and
Hugging Face namespaces. Revision pins and hashes verify their contents;
availability still depends on those repositories. Organization mirrors have
not been provisioned. A source mirror can be installed at the same pinned
commit without changing the adapter.

## Score a reasoning prefix

```python
from thinkbooster.scorers import StepScorerBELT

scorer = StepScorerBELT(
    checkpoint_repo="JingweiNi/BELT-Qwen3-1.7B-ReProbeInit",
    seed=0,
    device="cuda:0",
)
chat = [{"role": "user", "content": "What is 17 + 25?"}]
steps = ["17 + 20 = 37.\n\n", "37 + 5 = 42.\n\n"]
signal = scorer.score_prefix(chat, steps)
print(signal.beliefs)
print(len(signal.hidden))  # 768
```

Model loading is lazy. The first scoring call downloads one trained probe,
its reader template and the separately pinned base model. Checkpoint and
template SHA256 values are checked before loading. The backbone is
`Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
The default checkpoint revision is
`1ccc94337a9672dee94f79bc3c3a98a2f994220b`.

For candidate ranking, `score_candidates(chat, candidates, trajectory=steps)`
returns the chosen belief (`target="q"` by default). The detailed scoring
interface includes all four beliefs and checkpoint identity in its metadata.
Ranking by a belief is distinct from evaluating a fitted stopping policy.

`score_prefix` applies the chat template and tokenizes the joined step text
once. Character offsets map the step cuts to token ends. A token spanning a
cut belongs to the earlier step; a step entirely swallowed by that token adds
no new block. For exact replay of an existing tokenized bank, use
`scorer.runtime.score_token_ids(prompt_ids, completion_ids, block_ends)`.
`block_ends` contains strictly increasing,
exclusive offsets into completion tokens, including the current prefix end.
The original runtime applies its next-token feature alignment.

## Supply a fitted stopping policy

Reader checkpoints contain the probe and belief readouts. Their download does
not contain a fitted budget-specific stopping policy. Supply parameters and
a threshold selected using compatible policy-fit/development data.

`BeltStoppingPolicy.from_file(path)` reads a JSON object with these fields:

| Field | Meaning |
| --- | --- |
| `schema` | `thinkbooster-belt-policy/v1` |
| `checkpoint_sha256` | SHA256 of the trained `s1_best.pt` used to fit the policy |
| `feature_space` | `H` for hidden representation, `B` for belief logits |
| `weights`, `intercept` | Fitted linear parameters |
| `means`, `stds` | Policy-fit feature normalization, in coefficient order |
| `threshold` | Explicit decision-logit cutoff; stop when logit ≥ threshold |
| `length_scale` | Prefix-length denominator for five-input B policies; default 12000 |
| `logit_clip` | B belief clipping before conversion to logits; default `1e-4` |
| `hidden_dtype` | H feature storage convention, `float32` (default) or `float16` |

H consumes all 768 hidden coordinates in their saved order. B consumes
`logit(q), logit(v), logit(r), logit(stability)` in that order, optionally
followed by `prefix_tokens / length_scale`. Each coordinate is standardized
with its fitted mean and standard deviation before applying the weights.
Nonfinite parameters, nonpositive standard deviations, dimension errors and
checkpoint mismatches raise an error. Constructing `StrategyBELT` checks the
scorer's `checkpoint_sha256` against the policy before any generation. The
Hydra model factory performs this check before allocating vLLM; it fetches only
the pinned checkpoint manifest, without loading probe weights or a backbone.

Historical H policy fits read float16 representation dumps. Set
`hidden_dtype="float16"` when using their coefficients and normalization;
the policy rounds hidden coordinates to float16 before standardization.
New fits on `signal.hidden` use the default float32 convention.

For a selected fitted row and its matching normalization arrays, serialize
the policy as follows. The variables `selected_fit`, `normalization` and
`selected_logit_threshold` are outputs of your policy fitting and selection:

```python
import json
from dataclasses import asdict
from thinkbooster.belt import BeltStoppingPolicy

policy = BeltStoppingPolicy(
    checkpoint_sha256=signal.checkpoint_sha256,
    feature_space="H",
    weights=tuple(selected_fit["w"]),
    intercept=selected_fit["b"],
    means=tuple(normalization["mean"]),
    stds=tuple(normalization["std"]),
    threshold=selected_logit_threshold,
)
with open("belt-policy.json", "w") as stream:
    json.dump({"schema": "thinkbooster-belt-policy/v1", **asdict(policy)}, stream)
```

[BELT's policy recipes](https://github.com/EdisonNi-hku/BELT/blob/main/docs/RECIPES.md#policy-fitting-and-osp)
describe the original utility objective and selection conventions. Preserve
the selected coefficients, normalization, threshold and checkpoint identity
together. Mean coefficients reported in a paper are insufficient to reconstruct
a seed's fitted policy.

## Stop a ThinkBooster reasoning trajectory

```python
from thinkbooster.strategies import StrategyBELT

strategy = StrategyBELT(
    step_generator=step_generator,
    scorer=scorer,
    policy="belt-policy.json",
    max_steps=128,
    min_steps=1,
    query_stride=1,
)
result = strategy.generate_trajectory(chat)
print(result["trajectory"])
print(result["metadata"]["decisions"])
```

Use `VLLMStepGenerator` or a generator implementing both batched step and
answer APIs. The legacy HF generator's single-request API is not supported
by this strategy. The strategy generates one
candidate per active request and evaluates the policy on the configured query
cadence. A stop closes thinking and requests a final answer. Natural answers
are preserved. A step cap, context limit or failed/incomplete answer is
reported separately from a completed answer. Every decision records its
beliefs, prefix length, policy logit, threshold and checkpoint hash. Injected
`\n</think>\n\n` control text follows Qwen3's closing format and is recorded
separately from generated output tokens. `validity_scores` has one entry per
returned step: queried reasoning steps contain `q`; unqueried steps, injected
control text and final answers contain `None`.

A compatible custom scorer must expose `score_prefix(chat, trajectory)` and
`checkpoint_sha256`, so checkpoint identity can be checked before decoding.

The Hydra example uses a vLLM generation model on GPU 0 and the native HF
scoring model on GPU 1:

```bash
python scripts/run_tts_eval.py \
  --config-name experiments/belt/math500_qwen3_1p7b \
  strategy.policy_path=/absolute/path/to/belt-policy.json
```

The model and tokenizer revision are passed to vLLM. `scorer.seed` selects
the trained probe; `system.seed` controls generation. Choose a policy fitted
for that probe. A single-GPU setup can use `scorer.device=cuda:0` with a
generation memory budget that leaves room for the separate scoring model.

## Scoring cost

Each BELT query runs the separate HF Qwen3-1.7B model over the **entire prompt
and reasoning prefix**. This adapter does not reuse vLLM hidden states or a
KV cache across scoring calls. Chunking shares a KV cache within a single
forward only. Requests are scored one at a time, even when generation is
batched.

For `k` queries separated by `d` completion tokens and a prompt of `P` tokens,
the cumulative scoring input is approximately `k*P + d*k*(k+1)/2` tokens.
Thus processed prefix tokens grow quadratically with trace length at a fixed
query interval. The default `query_stride=1` queries after each eligible
reasoning step; with 50–300 token steps, long traces can require dozens of
full-prefix forwards. A larger stride reduces the number of scoring calls,
but changes the stopping opportunities: fit or validate the policy on the
chosen cadence.

`token_stats` and the evaluation runner's token-derived FLOP estimates cover
generation, **excluding BELT scoring**. The result records
`metadata.scoring_cost_in_token_stats=false`. Fewer generated tokens alone
do not establish lower latency or lower total compute for this adapter.
Measure scoring and generation together when assessing end-to-end cost.
Reusing vLLM's hidden-state capture requires a separate backend implementation
and validation of its layer, token and step-end alignment.

## Validation and reproduction scope

The adapter uses ThinkBooster's configured prompts and step boundaries.
Fit or validate policy parameters on that same interface before using them
for stopping. Published BELT paper readings use the original trajectory banks,
query grid and evaluation protocol; those numbers are not measurements of
this integration. The independent BELT repository retains the paper evidence.

Local logic tests require neither checkpoint downloads nor a GPU:

```bash
pytest tests/belt -m "not integration"
python tests/strategy_registry.py --validate
```

Opt-in CPU tests load a real published checkpoint and compare the adapter
with the original reader/readout on deterministic frozen features, including
all end markers and feature alignment. They also exercise the real pinned
Qwen tokenizer at step cuts produced by stripped generation stop strings:

```bash
BELT_TEST_CACHE=/path/to/hf-cache pytest tests/belt/test_native_reader.py \
  -m integration -o addopts= -v
```

This checks reader inference parity. It does not run generation or a benchmark.
