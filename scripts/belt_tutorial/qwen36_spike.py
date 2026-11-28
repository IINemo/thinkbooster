"""Cost spike for training a BELT probe on Qwen3.6-35B-A3B.

BELT's training data is a bank of sampled reasoning chains with a graded
forced answer at every fourth block end, and its trainer reads the backbone's
hidden states over every chain. This script times the three costs on one GPU
with vLLM, on a small sample of BELT-style training questions (DeepScaleR):

1. ``generate``: thinking chains under BELT's sampler (T=0.6, top-p 0.95,
   top-k 20) and token budget; generated tokens per second, chain lengths,
   accuracy.
2. ``forced``: a forced ``\\boxed{`` answer at every fourth block end up to
   block 128, 32 tokens each; seconds per forced answer and the prefix-cache
   hit rate (the hybrid linear-attention layers may limit prefix reuse).
3. ``prefill``: one full forward over each prompt plus chain with prefix
   caching reset, as a proxy for the hidden-state pass the trainer makes.

    python scripts/belt_tutorial/qwen36_spike.py --n 128 --out spike.json
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
REVISION = "c653bba46dfa06e9658dcc0b0dc87fe4685f14a2"
PROMPT = (
    "Answer the following question. Reply with the answer only. "
    "Answer enclosed with \\boxed{{}}.\n<Question>: {question}"
)
CLOSE = "\n</think>\n\n\\boxed{"


def boxed(text):
    k = text.rfind("\\boxed{")
    if k < 0:
        return None
    depth, out = 1, []
    for ch in text[k + 7 :]:
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            return "".join(out)
        out.append(ch)
    return None


def grade(pred, gold):
    if pred is None:
        return False
    try:
        from math_verify import parse, verify

        return bool(verify(parse("$" + gold + "$"), parse("$" + pred + "$")))
    except Exception:
        return pred.strip() == gold.strip()


def prefix_cache_stats(llm):
    hits = queries = 0.0
    try:
        for m in llm.get_metrics():
            if m.name.endswith("prefix_cache_hits"):
                hits += getattr(m, "value", 0) or 0
            if m.name.endswith("prefix_cache_queries"):
                queries += getattr(m, "value", 0) or 0
    except Exception:
        return None
    return {"hits": hits, "queries": queries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--budget", type=int, default=16384)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--block-cap", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    rows = list(load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train"))
    rows = [r for r in rows if re.fullmatch(r"-?\d+", str(r["answer"]).strip())]
    random.Random(0).shuffle(rows)
    rows = rows[: args.n]

    llm = LLM(
        model=MODEL,
        revision=REVISION,
        max_model_len=args.budget + 2048,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        seed=0,
    )
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(question=r["problem"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in rows
    ]
    report = {"model": MODEL, "revision": REVISION, "n": len(rows)}

    # 1. Chains.
    sp = SamplingParams(
        temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.budget, seed=0
    )
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    gen_s = time.perf_counter() - t0
    lens = [len(o.outputs[0].token_ids) for o in outs]
    texts = [o.outputs[0].text for o in outs]
    correct = [grade(boxed(t), str(r["answer"])) for t, r in zip(texts, rows)]
    capped = sum(o.outputs[0].finish_reason == "length" for o in outs)
    report["generate"] = {
        "seconds": gen_s,
        "generated_tokens": sum(lens),
        "tokens_per_second": sum(lens) / gen_s,
        "mean_tokens": sum(lens) / len(lens),
        "median_tokens": sorted(lens)[len(lens) // 2],
        "hit_budget": capped,
        "accuracy": sum(correct) / len(correct),
    }
    print(json.dumps(report["generate"]), flush=True)

    # 2. Forced answers at every stride-th block end of the thinking text.
    forced = []
    for p, t in zip(prompts, texts):
        think = t.split("</think>")[0]
        ends, pos = [], 0
        for block in think.split("\n\n")[:-1]:
            pos += len(block) + 2
            ends.append(pos)
        for b in range(args.stride, min(len(ends), args.block_cap) + 1, args.stride):
            forced.append(p + think[: ends[b - 1]] + CLOSE)
    before = prefix_cache_stats(llm)
    fp = SamplingParams(temperature=0.0, max_tokens=32, stop=["}"])
    t0 = time.perf_counter()
    fouts = llm.generate(forced, fp)
    forced_s = time.perf_counter() - t0
    after = prefix_cache_stats(llm)
    prompt_tokens = sum(len(o.prompt_token_ids) for o in fouts)
    report["forced"] = {
        "answers": len(forced),
        "per_chain": len(forced) / len(rows),
        "seconds": forced_s,
        "seconds_per_answer": forced_s / max(1, len(forced)),
        "prompt_tokens": prompt_tokens,
        "prefix_cache": (
            {
                "hits": after["hits"] - before["hits"],
                "queries": after["queries"] - before["queries"],
            }
            if before and after
            else None
        ),
    }
    print(json.dumps(report["forced"]), flush=True)

    # 3. Full-chain prefill with a cold prefix cache (hidden-state pass proxy).
    llm.reset_prefix_cache()
    full = [p + t for p, t in zip(prompts, texts)]
    t0 = time.perf_counter()
    pouts = llm.generate(full, SamplingParams(max_tokens=1))
    pre_s = time.perf_counter() - t0
    ptoks = sum(len(o.prompt_token_ids) for o in pouts)
    report["prefill"] = {
        "seconds": pre_s,
        "tokens": ptoks,
        "tokens_per_second": ptoks / pre_s,
    }
    print(json.dumps(report["prefill"]), flush=True)

    import torch
    import vllm

    report["gpu"] = torch.cuda.get_device_name(0)
    report["vllm"] = vllm.__version__
    args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
