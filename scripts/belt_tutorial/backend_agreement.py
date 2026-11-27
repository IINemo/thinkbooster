"""Compare BELT's vLLM feature backend with the HF backend on real decision points.

A never-stop run scored with ``scorer.backend=hf`` logs, at every decision
point, the four beliefs and the policy logit. This script rescores exactly
those prefixes (same chat, same step text) through the opt-in vLLM backend,
which reads layer-16 states from a vLLM engine, and reports how far beliefs and
logits move and how often stopping decisions change.

    python scripts/belt_tutorial/backend_agreement.py \
        --run <root>/runs/math500/natural/seed1 \
        --policy <root>/policies/natural.json \
        --thresholds 0 -2.04 --max-samples 100 --out agreement.json

Decision agreement is reported per decision point and, more to the point, as
the first stop of each trajectory: the block where each backend would stop.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def prompt_text(config, question):
    template = config.dataset.get("prompt_template") or "{question}"
    return template.format(question=question) if "{question}" in template else question


def first_stop(logits, threshold):
    for k, value in enumerate(logits):
        if value >= threshold:
            return k
    return None


def quantiles(values, qs=(0.5, 0.9, 0.99, 1.0)):
    values = sorted(values)
    if not values:
        return {}
    return {
        f"p{int(q * 100)}": values[min(len(values) - 1, int(math.ceil(q * len(values))) - 1)]
        for q in qs
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.0])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from omegaconf import OmegaConf
    from transformers import AutoConfig
    from vllm import LLM

    from thinkbooster.belt import BeltStoppingPolicy
    from thinkbooster.belt.runtime import BACKBONE, BACKBONE_REVISION
    from thinkbooster.belt.vllm_backend import BeltVLLMBackend
    from thinkbooster.scorers.step_scorer_belt import StepScorerBELT

    config = OmegaConf.load(args.run / ".hydra" / "config.yaml")
    policy = BeltStoppingPolicy.from_file(args.policy)
    samples = [
        s
        for s in json.loads((args.run / "results.json").read_text())
        if (s.get("metadata") or {}).get("decisions")
    ]
    if args.max_samples and len(samples) > args.max_samples:
        samples = random.Random(0).sample(samples, args.max_samples)

    llm = LLM(
        model=BACKBONE,
        revision=BACKBONE_REVISION,
        tokenizer_revision=BACKBONE_REVISION,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        seed=0,
        worker_extension_cls="thinkbooster.belt.vllm_worker.BeltHiddenStatesExtension",
    )
    backend = BeltVLLMBackend(llm)
    scorer = StepScorerBELT(
        feature_provider=backend,
        tokenizer=backend.get_tokenizer(),
        base_config=AutoConfig.from_pretrained(BACKBONE, revision=BACKBONE_REVISION),
        device="cuda:0",
        dtype="bfloat16",
    )

    points = []
    trajectories = []
    mismatched = []
    start = time.perf_counter()
    for sample in samples:
        chat = [{"role": "user", "content": prompt_text(config, sample["question"])}]
        steps = [s.get("raw_text") or s.get("text") for s in sample["steps"]]
        hf_logits, vllm_logits = [], []
        for decision in sample["metadata"]["decisions"]:
            signal = scorer.score_prefix(chat, steps[: decision["step"]])
            if signal.prefix_tokens != decision["prefix_tokens"]:
                # Not the same input as the logged HF decision; do not compare.
                mismatched.append((sample["index"], decision["block"]))
                hf_logits.append(decision["policy_logit"])
                vllm_logits.append(decision["policy_logit"])
                continue
            logit = policy.decision_logit(signal)
            hf_logits.append(decision["policy_logit"])
            vllm_logits.append(logit)
            points.append(
                {
                    "index": sample["index"],
                    "block": decision["block"],
                    "prefix_tokens": decision["prefix_tokens"],
                    "hf_beliefs": decision["beliefs"],
                    "vllm_beliefs": dict(signal.beliefs),
                    "hf_logit": decision["policy_logit"],
                    "vllm_logit": logit,
                }
            )
        trajectories.append(
            {"index": sample["index"], "hf": hf_logits, "vllm": vllm_logits}
        )
    elapsed = time.perf_counter() - start

    belief_err = [
        max(abs(p["hf_beliefs"][k] - p["vllm_beliefs"][k]) for k in p["hf_beliefs"])
        for p in points
    ]
    logit_err = [abs(p["hf_logit"] - p["vllm_logit"]) for p in points]
    long_err = [e for e, p in zip(logit_err, points) if p["prefix_tokens"] >= 4000]
    agreement = {}
    for t in args.thresholds:
        same_point = sum((p["hf_logit"] >= t) == (p["vllm_logit"] >= t) for p in points)
        same_stop = sum(
            first_stop(tr["hf"], t) == first_stop(tr["vllm"], t) for tr in trajectories
        )
        agreement[str(t)] = {
            "decision_points_agree": same_point / len(points),
            "first_stop_agree": same_stop / len(trajectories),
            "first_stop_disagreements": [
                {
                    "index": tr["index"],
                    "hf_stop_decision": first_stop(tr["hf"], t),
                    "vllm_stop_decision": first_stop(tr["vllm"], t),
                }
                for tr in trajectories
                if first_stop(tr["hf"], t) != first_stop(tr["vllm"], t)
            ],
        }
    report = {
        "run": str(args.run),
        "policy": str(args.policy),
        "samples": len(trajectories),
        "decision_points": len(points),
        "prefix_length_mismatches": mismatched,
        "scoring_seconds": elapsed,
        "max_abs_belief_error": quantiles(belief_err),
        "abs_logit_error": quantiles(logit_err),
        "abs_logit_error_prefix_ge_4000": quantiles(long_err),
        "agreement": agreement,
        "backend_stats": backend.stats,
    }
    args.out.write_text(json.dumps(report, indent=2))
    (args.out.with_suffix(".points.jsonl")).write_text(
        "".join(json.dumps(p) + "\n" for p in points)
    )
    print(json.dumps({k: v for k, v in report.items() if k != "agreement"}, indent=2))
    for t, a in agreement.items():
        print(
            f"threshold {t}: decision points agree {a['decision_points_agree']:.4f}, "
            f"first stop agrees {a['first_stop_agree']:.4f}"
        )


if __name__ == "__main__":
    main()
