# Training a BELT probe for Qwen3.6-35B-A3B: plan and estimates

Draft, 2026-11-27.

## Answer first

**Our GPUs can do it, slowly.** The FP8 model fits one 48 GB card, and so
does the trainer's backbone pass up to layer 24. On one card the whole job is
about **4 GPU-days** of compute (data about 3, probe training about 1) plus
**7 to 10 working days** of engineering. They overlap: chains generate
while the trainer and ThinkBooster are ported, so the calendar is about
**2 weeks** with one GPU, and a second GPU mostly buys slack (the data stage
drops from 3 days to 1.5). On the hardware side, generation is the bottleneck:
after 35 GB of weights a 48 GB card keeps only 14 chains in flight. The paper
trained on GH200s (96 GB usable), which avoids exactly that.

Suggested order: pool and generator (days 1 to 2) → chains and forced answers
on the GPUs (days 3 to 5) while the trainer is ported → probe training (day 6)
→ policies, ThinkBooster integration and evaluation (days 7 to 10).

**The training data needs no human markup.** Every label comes from the
model's own chains: a forced, automatically graded answer at every fourth
reasoning block, and the chain's final answer. Numbers marked **measured** come from spikes on our hardware
(`scripts/belt_tutorial/qwen36_spike.py`, `qwen36_hf_reader_spike.py`); the
rest are estimates and say what they assume.

## Goal

A BELT probe (Now, End, Later, Same) and fitted BELT-B / BELT-H stopping
policies for Qwen3.6-35B-A3B, run through ThinkBooster's `StrategyBELT`, and
evaluated out of distribution on MATH-500, Minerva Math and AIME 2025, the way
the tutorial page shows Qwen3-1.7B today.

## What the paper did (from the released checkpoints' `summary.json`)

| | Qwen3-1.7B | OLMo-3-7B-Think |
|---|---|---|
| Hardware | 1 × GH200 120 GB | 1 × GH200 120 GB |
| Training questions | 3,918 | 3,921 |
| Training chains (per question) | 31,248 (8) | 15,672 (4) |
| Validation / policy-selection chains | 2,554 / 10,648 | 1,278 / 3,804 |
| Token budget | 12,000 | 16,384 |
| Reader | layer 16 states, 2,048 → 768 | layer 18 states (one-layer scalar mix), 4,096 → 768, 19.1M parameters |
| Probe training per seed | 6.2 h (3 epochs) | 10.4 h (3 epochs) |

Probe training reruns the frozen backbone over every chain each epoch (no
feature cache). The paper does not record how long the chain banks and forced
answers took to generate.

## The model

Qwen3.6-35B-A3B (`Qwen3_5MoeForConditionalGeneration`): 40 layers, hidden size
2,048, 256 experts with 8 active (about 3B active parameters), three of every
four layers linear attention. About 70 GB in bf16; the official FP8 checkpoint
is 35 GB. The chat template opens `<think>` in the prompt (like OLMo-3); the
recommended thinking sampler is temperature 1.0, top-p 0.95, top-k 20.

## Pipeline

1. **Question pool.** Rebuild BELT's integer-answer math pool with its own
   builder (`iclr_build_math_pool`: DAPO-Math-17k, NuminaMath-1.5 hard,
   DeepScaleR), screened for overlap with every evaluation set. About 3,900
   training questions, 320 validation questions (160 for policy fitting, 160
   for development) and a policy-selection set, as in the OLMo recipe.
2. **Training data (the "markup").** No human labelling: every label comes
   from the model's own chains and an automatic grader.
   - Four sampled chains per question at the model's sampler, 16k-token budget.
   - At every fourth blank-line block end up to block 128, close thinking and
     force a `\boxed{` answer of at most 32 tokens; grade it strictly against
     the reference.
   - Natural completion; when a closed chain has no parseable answer, elicit
     one (BELT's natural-answer fallback).
   - Labels per decision point: **Now** = forced answer correct; **End** = the
     chain's final answer correct; **Later** = a strictly later forced answer
     or the final answer correct; **Same** = forced answer equals the final
     answer (masked when the final answer cannot be parsed).
3. **Probe training.** BELT's trainer (`single_cot_train`) with the OLMo
   reader shape (one mid-depth layer's states, RMS-normalized, into a 768-d,
   2-layer, 8-head reader, random init), 3 epochs, select by validation loss. Feature layer: a
   full-attention layer near 60% depth (layer 24 of 40, the analogue of
   Qwen's 16/28 and OLMo's 18/32), confirmed by a cheap linear probe sweep
   over a few layers on the validation chains. One seed first; seeds 1 and 2
   if time allows.
4. **Policies.** BELT-B and BELT-H at 17 prices on the 160 policy-fit
   questions (float64 L-BFGS, CPU, minutes).
5. **ThinkBooster integration and evaluation.** Generalize the BELT runtime
   beyond its Qwen3-1.7B pins, add the reader shape, capture states in the
   vLLM backend for this architecture, check parity against the trainer's own
   scorer, then run MATH-500, Minerva and AIME and add the model to the page.

## Engineering work

| Piece | Work | Estimate |
|---|---|---|
| Question pool | Run BELT's pool builder, screen overlap with MATH-500, Minerva, AIME, OlympiadBench, GPQA | 0.5 day |
| BELT generator and labeller | A third prompt contract (Qwen3.6 template, `<think>` opened in the prompt), vLLM 0.31 backend, sharded runs | 1 to 1.5 days |
| BELT trainer | Load `Qwen3_5MoeForCausalLM` truncated to the feature layer, FP8 weights dequantized to bf16, feature contract; a layer sweep with linear probes | 1.5 to 2 days |
| Policies | BELT's fitting code on the new representation | 0.5 day |
| ThinkBooster, HF backend | Generalize the runtime's Qwen3-1.7B pins, new reader shape, a separate vLLM 0.31 environment; the HF scorer on a second GPU (it does not fit beside the 35 GB model) | 1.5 to 2 days |
| ThinkBooster, vLLM backend (optional) | Port the hidden-state capture to `qwen3_5_moe` in vLLM 0.31, so one GPU serves both | 2 to 3 days |
| Parity, evaluation, page | Reader parity against the trainer's scorer; MATH-500, Minerva, AIME runs; tutorial page | 1 to 1.5 days |
| **Total** | | **7 to 10 days** (8 to 13 with the vLLM backend) |

## Compute

### Measured on our hardware

One RTX 6000 Ada 48 GB (Artem-3), Qwen3.6-35B-A3B-FP8, vLLM 0.31, 128
DeepScaleR training questions, BELT's protocol (16,384-token budget, a forced
answer at every fourth block end up to block 128):

| Stage | Measured |
|---|---|
| Chain generation | **837 tokens/s**; mean chain 9,841 tokens, median 12,192; **43% of chains hit the 16k budget**; 50% correct |
| Concurrency | 33.4 GiB of weights leave 6.8 GiB of cache: at most **14 chains** of 18k tokens in flight |
| Forced answers | **0.106 s each**, 9.4 per chain (Qwen3.6 writes long blocks, about 260 tokens), so about 1 s per chain; prefixes are reused from cache |
| Full-chain prefill in vLLM | 14,046 tokens/s |
| Trainer backbone pass, Transformers, truncated, FP8 dequantized to bf16 | 16 layers: **13,700 tokens/s**, peak 31 GB; 24 layers: **9,000 tokens/s**, peak 43 GB |

The FP8 weights load in Transformers only as bf16 here: its native FP8 path
fetches kernel code from an unverified Hub repository, which we did not
enable.

### Extrapolated to the OLMo-scale recipe

About 3,900 training questions with 4 chains each, plus validation and
selection chains: **about 20,700 chains of 9,800 tokens, 204M tokens.**

| Stage | One GPU | Two GPUs |
|---|---|---|
| Chains (204M tokens at 837 tokens/s) | 68 h | 34 h |
| Forced answers (194k at 0.106 s) | 6 h | 3 h |
| Probe training, layer 24, 3 epochs (156M tokens per epoch at 9,000 tokens/s, plus about 30% trainer overhead and validation) | 20 h per seed | 20 h (one seed per GPU) |
| Policy fitting | minutes (CPU) | minutes |
| Evaluation in ThinkBooster (MATH-500, Minerva, AIME: never-stop runs, forced answers, live runs) | about 8 h | about 4 h |
| **Total for one seed** | **about 4.2 GPU-days** | **about 2.6 days wall** |

Halving the pool (about 2,000 questions) halves the data stage. Each extra
seed adds about 20 GPU-hours. For comparison, the paper's OLMo probe took
10.4 h per seed on a GH200.

### Decisions this needs

1. **Token budget.** At 16k, 43% of chains on these hard questions are cut
   off; BELT does not rescue capped chains, so their End label is "wrong".
   A 32k budget roughly doubles the data cost and halves concurrency again.
   Proposal: 16k as in the OLMo recipe, with a slightly easier pool (more
   DAPO and NuminaMath, less DeepScaleR), and report the cap rate.
2. **Sampler.** The spike used BELT's T = 0.6; Qwen3.6 recommends T = 1.0
   for thinking. The probe should be trained under the sampler we deploy.
3. **Feature layer.** Layer 24 (60% depth, like the paper's choices) fits
   with 43 GB peak; layer 16 is 50% faster and leaves headroom. A linear
   probe sweep over layers 12 to 28 on the validation chains decides.
4. **GPUs.** Two cards halve the calendar. An 80 GB or GH200 card (the
   paper used CSCS GH200s) would raise generation throughput several times.

## Hardware and storage (checked 2026-11-27)

| Host | GPUs | State for this work |
|---|---|---|
| Artem-3 | 2 × RTX 6000 Ada 48 GB | GPU 0 free (used for the spikes); GPU 1 another user's training |
| Artem-1 | 2 × RTX 5880 Ada 48 GB | GPU 1 runs the BELT tutorial queue (GPQA, AIME, HumanEval+); GPU 0 broken ("busy or unavailable") |
| Artem-2, Artem-4 | 2 × RTX 5880 Ada each | both GPUs in use by another user |
| Artem-5 | 2 GPUs | driver down (`nvidia-smi` fails) |
| GB10-01 / GB10-02 | 128 GB unified | 01 unreachable; 02 has 110 of 119 GB in use |
| Artem-6 to 9 | shared pool | assigned to other projects; the 2-VM cap is reached |

**Memory is not the blocker.** The FP8 checkpoint (35 GB) runs in vLLM on one
48 GB card. For training, only the layers up to the feature layer are needed
(about 60% of the weights for layer 24 of 40).

**Storage is.** Our per-user quota is 1 TB per host and is nearly full: about
50 GB free on Artem-3 after the model download, about 47 GB on Artem-1. Chains
and labels are a few GB. A cache of one layer's hidden states for every token
of every chain (2,048 values in bf16 per token, about 166M tokens) would be
about 680 GB, so the trainer must recompute the backbone each epoch, as the
paper's runs did, unless old project data is cleared.

**Environment.** Both hosts' home directories are over quota and Artem-3's
root disk is full; runs must keep HOME, caches and temp files under
`/mnt/data`. A separate vLLM 0.31 environment (`venvs/belt-qwen36` on
Artem-3) runs the model; ThinkBooster itself pins vLLM 0.12.

## Risks

- **Throughput of a hybrid MoE on Ada GPUs.** Measured below; the paper's
  GH200 is several times faster per GPU.
- **Prefix reuse for forced answers.** Each forced answer branches off a
  chain prefix. With linear-attention layers vLLM may not reuse that prefix,
  which would make labelling cost scale with prefix length. Measured below.
- **Transformers MoE speed for the trainer.** The trainer runs the backbone
  in Transformers, whose MoE path is slower than vLLM's. Measured below.
- **FP8 states.** The probe would be trained and served on the FP8 model, so
  train and serve match, but its numbers are not comparable to a bf16 probe.
- **Porting ThinkBooster to vLLM 0.31.** The BELT vLLM backend's state
  capture is written for vLLM 0.12 and dense Qwen3. The HF backend avoids it
  but needs a second copy of the model (truncated: about 21 GB in FP8).
- **GPU availability.** With one free GPU, every stage runs serially.
