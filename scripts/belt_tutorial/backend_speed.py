"""Time BELT scoring per decision point with the HF and the vLLM backend.

Both scorers live in one process on one GPU. For every logged decision point
of a never-stop run, the same prefix is scored by both backends back to back,
in random order, with CUDA synchronized around each call, so both see the same
background load. This measures scoring alone:

* HF recomputes the whole prompt + reasoning prefix through a separate
  transformers backbone at every query.
* vLLM reads layer-16 states from a vLLM engine. Here no generation ran in this
  engine, so each new prefix is filled by a repair prefill (APC reuses the
  earlier prefix of the same trajectory). In a live run, generation captures
  these features as it decodes, so this is an upper bound on vLLM query cost.

    python scripts/belt_tutorial/backend_speed.py \
        --run <root>/runs/minerva/natural/seed1 --max-samples 40 --out speed.json
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def prompt_text(config, question):
    template = config.dataset.get("prompt_template") or "{question}"
    return template.format(question=question) if "{question}" in template else question


def summary(values):
    values = sorted(values)
    n = len(values)
    if not n:
        return {}
    return {
        "n": n,
        "mean_ms": 1000 * sum(values) / n,
        "p50_ms": 1000 * values[n // 2],
        "p90_ms": 1000 * values[min(n - 1, int(0.9 * n))],
        "total_s": sum(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.18)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    import torch
    from omegaconf import OmegaConf
    from transformers import AutoConfig
    from vllm import LLM

    from thinkbooster.belt.runtime import BACKBONE, BACKBONE_REVISION
    from thinkbooster.belt.vllm_backend import BeltVLLMBackend
    from thinkbooster.scorers.step_scorer_belt import StepScorerBELT

    config = OmegaConf.load(args.run / ".hydra" / "config.yaml")
    samples = [
        s
        for s in json.loads((args.run / "results.json").read_text())
        if (s.get("metadata") or {}).get("decisions")
    ]
    rng = random.Random(0)
    if len(samples) > args.max_samples:
        samples = rng.sample(samples, args.max_samples)

    llm = LLM(
        model=BACKBONE,
        revision=BACKBONE_REVISION,
        tokenizer_revision=BACKBONE_REVISION,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        seed=0,
        worker_extension_cls="thinkbooster.belt.vllm_worker.BeltHiddenStatesExtension",
    )
    backend = BeltVLLMBackend(llm)
    scorers = {
        "vllm": StepScorerBELT(
            feature_provider=backend,
            tokenizer=backend.get_tokenizer(),
            base_config=AutoConfig.from_pretrained(
                BACKBONE, revision=BACKBONE_REVISION
            ),
            device="cuda:0",
        ),
        "hf": StepScorerBELT(device="cuda:0"),
    }

    # Warm both up (loading, kernels) on one prefix before timing.
    first = samples[0]
    warm_chat = [{"role": "user", "content": prompt_text(config, first["question"])}]
    warm_steps = [s.get("raw_text") or s.get("text") for s in first["steps"]][:8]
    for scorer in scorers.values():
        scorer.score_prefix(warm_chat, warm_steps)

    times = {"hf": [], "vllm": []}
    rows = []
    for sample in samples:
        chat = [{"role": "user", "content": prompt_text(config, sample["question"])}]
        steps = [s.get("raw_text") or s.get("text") for s in sample["steps"]]
        for decision in sample["metadata"]["decisions"]:
            prefix = steps[: decision["step"]]
            order = ["hf", "vllm"]
            rng.shuffle(order)
            row = {"index": sample["index"], "prefix_tokens": decision["prefix_tokens"]}
            for name in order:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                scorers[name].score_prefix(chat, prefix)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                times[name].append(elapsed)
                row[name + "_s"] = elapsed
            rows.append(row)

    buckets = {}
    for lo, hi in ((0, 1000), (1000, 4000), (4000, 8000), (8000, 10**9)):
        sel = [r for r in rows if lo <= r["prefix_tokens"] < hi]
        buckets[f"{lo}-{hi if hi < 10**9 else 'max'}"] = {
            "hf": summary([r["hf_s"] for r in sel]),
            "vllm": summary([r["vllm_s"] for r in sel]),
        }
    report = {
        "run": str(args.run),
        "samples": len(samples),
        "decision_points": len(rows),
        "gpu": torch.cuda.get_device_name(0),
        "hf": summary(times["hf"]),
        "vllm": summary(times["vllm"]),
        "by_prefix_tokens": buckets,
        "vllm_backend_stats": backend.stats,
        "note": "vLLM queries are cold repair prefills (no generation in this engine)",
    }
    args.out.write_text(json.dumps(report, indent=2))
    args.out.with_suffix(".rows.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    print(
        json.dumps({k: report[k] for k in ("decision_points", "hf", "vllm")}, indent=2)
    )
    for name, b in buckets.items():
        print(name, {k: round(v.get("mean_ms", 0), 1) for k, v in b.items()})


if __name__ == "__main__":
    main()
