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
once. Block ends come from the text, by BELT's own rule, not from how it was
cut into steps: a block ends at a blank line (`\n[ \t]*\n`) that follows
non-whitespace text, and the end marker goes after the separator. The prefix
end is always the last block end; whitespace after a blank line belongs to the
block it closes. Character offsets map these ends to tokens, and a token
spanning an end (`.\n\n`) belongs to the block it closes. For exact replay of an existing tokenized bank, use
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
    min_steps=2,
    max_blocks=128,
    query_stride=1,
    max_steps=1024,
)
result = strategy.generate_trajectory(chat)
print(result["trajectory"])
print(result["metadata"]["decisions"])
```

Use `VLLMStepGenerator` or a generator implementing both batched step and
answer APIs, in thinking mode: the strategy rejects generators with
`thinking_mode=False` or `disable_thinking_mode=True`, because the reader
scores thinking prompts and a stop must close an open thinking block. The legacy HF generator's single-request API is not supported
by this strategy. The strategy generates one
candidate per active request, one BELT block at a time, as BELT itself does:
each step stops on `"\n\n"` with the stop string kept in the text, so the
next prompt is exactly what was sampled. The generator must therefore accept
per-call `stop_tokens_override`, `include_stop_str_in_output` and
`truncate_at_sentence`, and must have `min_step_tokens=0`; vLLM ignores stop strings before `min_tokens`, which
would merge short blocks. The policy is queried only when the thinking text
ends on a new block end, at block index `b` with
`min_steps <= b <= max_blocks` and `b % query_stride == 0`. Blocks are counted
as BELT counts them; block 1 is never a decision under the default
`min_steps=2`, and blocks past `max_blocks` are generated without queries. A
block longer than `max_step_tokens` is kept whole (not trimmed to a sentence)
and finished by the next round; no decision is made at that cut. `max_steps` caps generation rounds. A stop closes thinking and requests a final answer. Natural answers
are preserved. A step cap, context limit or failed/incomplete answer is
reported separately from a completed answer. Every decision records its
block index, step index, beliefs, prefix length, policy logit, threshold and
checkpoint hash. Injected
`\n</think>\n\n` control text follows Qwen3's closing format and is recorded
separately from generated output tokens. `validity_scores` has one entry per
returned step: queried reasoning steps contain `q`; unqueried steps, injected
control text and final answers contain `None`.
The evaluation runner preserves stopping reasons and decision metadata in
progressive checkpoints, batch JSONL files and final results. Other strategies'
metadata is not written to result files. On resume, progressive-checkpoint rows
are discarded and those samples are regenerated; final rows are kept.

A compatible custom scorer must expose `score_prefix(chat, trajectory)` and
`checkpoint_sha256`, so checkpoint identity can be checked before decoding.

The Hydra example uses a vLLM generation model on GPU 0 and the native HF
scoring model on GPU 1:

```bash
python scripts/run_tts_eval.py \
  --config-path "$PWD/config" \
  --config-name experiments/belt/math500_qwen3_1p7b \
  strategy.policy_path=/absolute/path/to/belt-policy.json
```

The model and tokenizer revision are passed to vLLM. `scorer.seed` selects
the trained probe; `system.seed` controls generation. Choose a policy fitted
for that probe. A single-GPU setup can use `scorer.device=cuda:0` with a
generation memory budget that leaves room for the separate scoring model.

## Opt-in shared vLLM backend

**Numerical status:** the shared backend currently fails the fixed-input HF
parity checks described below. Use `scorer.backend=hf` when you need the
HF-defined backbone features.

The default remains `scorer.backend=hf`. To use one generation backbone and
the native BELT reader on GPU 0:

```bash
python scripts/run_tts_eval.py \
  --config-path "$PWD/config" \
  --config-name experiments/belt/math500_qwen3_1p7b_vllm \
  strategy.policy_path=/absolute/path/to/belt-policy.json
```

This initial backend targets **vLLM 0.12, the pinned Qwen3-1.7B revision,
TP=1, PP=1, eager execution, bfloat16 weights and KV cache, no quantization,
and one completion per request**. It requires `kv_cache_dtype=auto` and a
frozen base model: LoRA configuration/requests, speculative decoding and
prompt embeddings are rejected, including through the direct Python API.
Cache identities contain actual token IDs, so adapter-specific states and
synthetic token IDs from prompt embeddings are outside the supported scope.
The factory rejects unsupported backbones, revisions and backend settings
before allocating the engine. It uses its own worker extension; the generic
upstream hidden-state extension is unchanged.

On 2026-10-18, the five fixed-prefix cases were checked on one RTX 4090 with
Python 3.11.16, PyTorch 2.9.0+cu128, Transformers 4.56.2 and vLLM 0.12.0.
The reader used the default seed-0 ReProbe-initialized checkpoint. The
1238-token case exceeded all four predeclared numerical limits:

| Quantity | Measured error | Limit |
| --- | ---: | ---: |
| Layer-16 features, relative L2 | 0.024634 | 0.020000 |
| Reader H, relative L2 | 0.022056 | 0.020000 |
| Beliefs, maximum absolute error | 0.011676 | 0.010000 |
| Fixed fixture policy logit, absolute error | 0.062576 | 0.050000 |

The remaining four fixed-prefix cases passed those limits. Warm-cache equality,
generation-feature reuse without query repair, and the generator/scorer/strategy
smoke checks passed. The subsequent HF replay of generated tokens was not run
because the preceding numerical gate failed. The cases, HF SDPA reference and
limits remain unchanged. This is software validation with an unfitted policy
fixture; it does not measure stopping quality. A separate cold replay retained
the long-input discrepancy. Layer-0 input and normalized input matched exactly;
the first observed difference appeared in the QKV projections. An isolated
CUDA replay with identical inputs and weights reproduced HF's separate Q/K/V
projections and vLLM's packed projection exactly. Disabling BF16 reduced-precision
reduction did not change the packed result. This identifies the first difference,
but does not establish a complete cause or a numerical fix.

The worker captures the full residual **at the input to zero-based decoder
layer 16**, equivalent to HF `hidden_states[16]`, with width 2048. In vLLM's
Qwen3 implementation this is `hidden_states + residual`, copied before the
fused normalization mutates either branch. It is not the bare MLP output.
Captures retain request IDs and absolute token positions; uncertain attribution
raises an error. Independent GPU copies remain bounded to 131072 captured
tokens per generation call and transfer to float32 CPU arrays on the flush
RPC. The transport uses a versioned little-endian float32 byte payload with
explicit dimensions; the frontend checks byte length, capture bounds and finite
values before making owned CPU arrays. It requires no insecure serialization
setting. Leave GPU memory room for these copies and the small native reader.

The example uses `generation.checkpoint_batch_size=4`. The runner accepts at
most `min(16, 131072 // max_model_len)` (4 at the default 32768-token context),
so a batch of full-length requests always fits the capture buffer. Raw
generation batches must fit the backend's `max_cache_entries` (default 16);
direct Python callers can explicitly raise that constructor limit when memory
permits. Before capture reset or generation, the backend rejects a conservative
sum over requests of `min(prompt tokens + max_tokens, max_model_len)` above
131072. Text uses the engine
tokenizer; explicit token IDs and per-request sampling parameters are supported.
Omitted parameters come from `LLM.get_default_sampling_params()`. Unbounded
`max_tokens` and unsupported prompt schemas are rejected; optional prompt
truncation does not reduce this conservative estimate. There is no automatic
microbatching. Unexpected recomputation or stop-abort lag can still overflow:
the worker stops capturing within the bound, then reports an error through the
flush utility RPC and releases buffers. It does not raise capacity errors inside
the model hook or return partial features. Failed overflow calls report observed
and retained rows in the error; successful-call capture totals do not estimate
uncaptured work from those failures.

The feature cache matches actual prompt/completion token IDs. Causal prefixes
can share features only while their token IDs match exactly. Retokenizing
joined step text can change a tail token even when the text looks identical;
changed tokens and later positions cannot reuse the previous capture. The
cache retains the first cached row for each exact causal token prefix while
an owner remains cached; overlapping captures fill only missing positions.
Longer entries inherit those rows before bounded-cache eviction. Evicting all
owners or resetting the cache permits a new measurement. Metadata records
`feature_cache_ownership=first_cached_causal_prefix`; capture counters include
all measured rows even when an existing owner is retained. The
last generated token normally has no captured feature yet. The reader retains
its canonical next-token alignment and consumes states for `full_ids[:-1]`.
If frontend stop-string handling allows extra decode work before abort, a
captured last returned token's row is retained for later exact extensions.
Rows beyond the returned sequence belong to unreturned tokens and are discarded
from the feature cache.

When normal generation capture does not cover that exact token sequence,
the provider requests a deterministic `max_tokens=1` prefill using explicit
`prompt_token_ids`, without stop strings. Known exact prefixes can reuse APC.
Unknown APC gaps trigger one prefix-cache reset and one cold retry; remaining
gaps raise an error. Starting a strategy batch clears both feature ownership
and APC. Scorer cleanup never destroys the shared generation engine.
An explicit `reset_hs_step_cache(reset_prefix_cache=False)` retains APC after
dropping feature ownership and can therefore require repairs; the strategy
uses the default `True`.

`BeltRuntime.from_pretrained(feature_provider=backend, tokenizer=tokenizer,
base_config=config)` and `StepScorerBELT` accept explicit injection too. This
path loads the original head and trained readout without an HF backbone.
Omitting `base_config` loads only the pinned `AutoConfig`. The Hydra factories
bind the generator's provider and tokenizer before lazy scorer loading.

Result metadata includes `scoring_backend`, `backend_stats`, and
`backend_stats_scope=strategy_batch`. Counters cover generation capture,
feature queries, cache hits, repair calls, actual computed/captured repair rows,
requested repair-prefill lengths, prefix-cache resets and timings. Actual
repair rows include the last prompt position even though the reader does not
consume it. `generation_captured_tokens`, `repair_tokens` and
`repair_captured_tokens` include all measured rows before filtering;
`discarded_overrun_rows` counts the discarded subset, rather than extra work to
add to those totals. These are shared batch totals, not per-sample work estimates.
`backend.metadata` also describes the captured layer and feature format.

CPU tests cover residual reconstruction and mutation isolation, request/position
attribution, exact-token cache merging, retokenized tails, bounded gap repair,
generator routing, reader-only loading and native reader parity. GPU validation
is a separate check; these CPU tests do not establish backbone numerical parity
or any latency or utilization benefit. There is no universal speedup claim:
capture copies, the reader and repair prefills all cost time and memory.

## Scoring cost

With the default HF backend, each BELT query runs the separate Qwen3-1.7B model over the **entire prompt
and reasoning prefix**. This adapter does not reuse vLLM hidden states or a
KV cache across scoring calls. Chunking shares a KV cache within a single
forward only. Requests are scored one at a time, even when generation is
batched.

For `k` queries separated by `d` completion tokens and a prompt of `P` tokens,
the cumulative scoring input is approximately `k*P + d*k*(k+1)/2` tokens.
Thus processed prefix tokens grow quadratically with trace length at a fixed
query interval. The default `query_stride=1` queries after each eligible
block; blocks average about 40 tokens, so long traces can require a hundred
full-prefix forwards. A larger stride reduces the number of scoring calls,
but changes the stopping opportunities: fit or validate the policy on the
chosen cadence.

`token_stats` and the evaluation runner's token-derived FLOP estimates cover
generation, **excluding BELT scoring**. The result records
`metadata.scoring_cost_in_token_stats=false`. The sample metrics JSONL also
records `scoring_cost_in_token_stats=false`. Fewer generated tokens alone do
not establish lower latency or lower total compute for this adapter.
Measure scoring and generation together when assessing end-to-end cost.
The opt-in vLLM backend above reuses captured features where token identity
and coverage permit. Its reader and repair work are likewise excluded from
generation token statistics; report backend counters alongside end-to-end cost.

## Validation and reproduction scope

The adapter steps and scores on BELT's blank-line blocks, but uses
ThinkBooster's configured prompts and answer phase. Fit or validate policy parameters on that same interface before using them
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
Qwen tokenizer at blank-line block ends, and check the block rules against
BELT's own `rl_controller` on random texts (BELT on `PYTHONPATH`):

```bash
BELT_TEST_CACHE=/path/to/hf-cache pytest tests/belt/test_native_reader.py \
  tests/belt/test_blocks.py -o addopts= -v
```

This checks reader inference parity. It does not run generation or a benchmark.

The opt-in [GPU validation helper](../../tests/belt/validate_vllm_gpu.py) checks
the backbone residuals as well. It is not collected by pytest. With the adapter
and native reader installed, run these commands on a CUDA GPU in **three
separate Python processes**. Each backbone is released before the next phase:

```bash
belt_snapshot=/path/to/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
python tests/belt/validate_vllm_gpu.py hf \
  --model "$belt_snapshot" --probe-cache /path/to/probe-cache \
  --output ./belt-gpu-validation
python tests/belt/validate_vllm_gpu.py vllm \
  --model "$belt_snapshot" --probe-cache /path/to/probe-cache \
  --output ./belt-gpu-validation
python tests/belt/validate_vllm_gpu.py hf-generated \
  --model "$belt_snapshot" --probe-cache /path/to/probe-cache \
  --output ./belt-gpu-validation
```

The helper requires an existing local snapshot with that exact revision and
`vllm==0.12.0`. All phases record software/GPU identity and the real pinned
seed-0 probe checksum; runtime loading verifies the probe/template hashes.
Content SHA256 values identify the actual imported public runtime/helper
sources and native reader runtime, including copies without `.git`; optional
Git revision and dirty status are also recorded. Reference phases reject
mismatched source content, software, GPU model, settings or tokenization.
Five fixed prefixes cover successive joined steps, stripped-stop token merges,
Unicode and a longer trace. Predeclared limits are relative L2 ≤ 0.02 for
backbone features and reader representations, maximum belief error ≤ 0.01,
and fixture-policy logit error ≤ 0.05. Keep those cases, coefficients and limits
fixed when interpreting a run.

The vLLM phase saves four actual 64-token continuations, their captured
`[T-1,2048]` features, and trained reader representations, beliefs and fixture
logits. Saved cases retain the original chat prompt and all prior step ends;
the generation input length is recorded separately. Capture queries and reader
scoring must reuse those features with **zero repairs**. This phase reports
`overall_status=pending_hf_generated` rather than final parity success.

The final `hf-generated` process forwards the **same saved token IDs** through
HF, frees that backbone, then runs the injected native reader. It compares
both full-sequence features and the 63 actual decode rows after the generation
input, excluding the canonically uncaptured final token. Decode rows use the
same 0.02 relative-L2 limit independently, so prefill rows cannot dilute decode
error. Reader representations, beliefs and fixture logits use the same limits
above. No sampled-text equality is required.

The helper also tests warm feature reuse, batched generation capture, and a
short real `VLLMStepGenerator` → `StepScorerBELT` → `StrategyBELT` trajectory.
It forbids HF backbone/runtime loading during injected scoring. Its policies
are explicitly labeled **unfitted software fixtures** bound to the real probe
checksum: one exercises logit parity; a zero-weight policy forces a stop at the
first eligible query. The smoke must actually query the reader and exercise
answer generation. Completion/truncation flags and generation, cache and repair
counters are reported; no answer-accuracy or fitted-policy claim is made.

Optional vLLM-phase flags `--max-num-batched-tokens` (default 8192),
`--max-num-seqs` (default 8), and `--disable-prefix-caching` exercise chunk
boundaries and request attribution. Chunked prefill is always enabled, and
`vllm.json` records the effective engine arguments. For example, use
`--max-num-batched-tokens 128 --disable-prefix-caching` for a smaller prefill
chunk without APC. Use a separate output directory per configuration and copy
the three HF reference artifacts (`cases.json`, `hf_features.pt`, `hf.json`)
there before running its vLLM phase, or rerun HF in that directory.

The reports are `hf.json`, `vllm.json`, and `generated_parity.json`. Only the
last report can establish final `overall_status=passed`; it also requires all
preceding vLLM checks to pass. Artifact hashes bind it to the exact reference,
generation cases and feature files. Rerunning an upstream phase invalidates
downstream reports, preventing stale successes. Failures exit nonzero and write
a phase failure JSON when preflight has completed. Timings are diagnostic
measurements of these software checks and do not establish a speedup.
The standalone helper always shuts down its owned vLLM engine, including after
validation errors, and writes its successful vLLM report after shutdown. A
secondary cleanup failure preserves the original validation exception.
