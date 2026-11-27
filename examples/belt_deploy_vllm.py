"""Deploy Qwen3-1.7B on vLLM with BELT early stopping, and see what it saves.

The engine loads the model into vLLM once, together with the BELT probe and a
fitted stopping policy. Every fourth reasoning block BELT reads the model's
beliefs about its own answer; when the policy says the answer will not get
better, it closes ``</think>`` and the model writes its final answer, so the
remaining reasoning is never generated.

This script answers the same MATH-500 problems twice on one engine, with BELT
and with no early stopping, and prints the reasoning tokens and accuracy of
both.

    # one-time: a policy from the paper's seed-0 fit (see the BELT tutorial page)
    python scripts/belt_tutorial/make_policy.py --sample-policy belt_h_r_seed0.json \\
        --lambda-index 7 --threshold 0 --out belt-policy.json

    python examples/belt_deploy_vllm.py --policy belt-policy.json --n 32
    python examples/belt_deploy_vllm.py --policy belt-policy.json --backend vllm

Serving the same engine behind an OpenAI-compatible API:

    BELT_POLICY_PATH=$PWD/belt-policy.json python service_app/main.py

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8001/v1", api_key="unused")
    r = client.chat.completions.create(
        model="Qwen/Qwen3-1.7B",
        messages=[{"role": "user", "content": "What is 17 * 24?"}],
        extra_body={"tts_strategy": "belt"},
    )
    print(r.choices[0].message.content, r.choices[0].tts_metadata["belt"])
"""

import argparse
import json
import time

# A threshold no stop score reaches: the model thinks until it stops itself.
NEVER = 1e9


def grade(pred, gold):
    if pred is None:
        return False
    try:
        from math_verify import parse, verify

        return bool(verify(parse("$" + gold + "$"), parse("$" + pred + "$")))
    except Exception:
        return pred.strip() == gold.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="fitted BELT policy JSON")
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    parser.add_argument("--n", type=int, default=32, help="MATH-500 problems")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--out", help="optional JSON with per-problem results")
    args = parser.parse_args()

    from datasets import load_dataset

    from thinkbooster.belt.engine import BeltEngine

    rows = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))[: args.n]
    questions = [r["problem"] for r in rows]

    t0 = time.perf_counter()
    engine = BeltEngine(
        args.policy,
        backend=args.backend,
        threshold=args.threshold,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(f"engine ready in {time.perf_counter() - t0:.0f} s ({args.backend} backend)")

    runs = {}
    for name, threshold in (("BELT", None), ("no early stopping", NEVER)):
        t0 = time.perf_counter()
        results = engine.generate(questions, threshold=threshold)
        runs[name] = {"seconds": time.perf_counter() - t0, "results": results}

    print(f"\n{len(rows)} MATH-500 problems, Qwen3-1.7B")
    print(
        f"{'':20} {'accuracy':>9} {'thinking tokens':>16} {'stopped early':>14} {'time':>7}"
    )
    for name, run in runs.items():
        res = run["results"]
        acc = sum(grade(r.answer, row["answer"]) for r, row in zip(res, rows)) / len(
            res
        )
        think = sum(r.thinking_tokens for r in res) / len(res)
        early = sum(r.stopped_early for r in res) / len(res)
        print(
            f"{name:20} {acc:9.1%} {think:16,.0f} {early:14.0%} {run['seconds']:6.0f}s"
        )
    belt = sum(r.thinking_tokens for r in runs["BELT"]["results"])
    full = sum(r.thinking_tokens for r in runs["no early stopping"]["results"])
    print(f"\nBELT used {1 - belt / max(full, 1):.0%} fewer reasoning tokens.")
    print("The two runs sample independently, so per-problem lengths differ by chance.")

    first = runs["BELT"]["results"][0]
    print(f"\nExample: {questions[0][:100]}")
    print(
        f"  answer {first.answer!r}, {first.thinking_tokens} thinking tokens, "
        f"stopped early: {first.stopped_early} (decision {first.stop_decision})"
    )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    name: [
                        {
                            "answer": r.answer,
                            "gold": row["answer"],
                            "thinking_tokens": r.thinking_tokens,
                            "stopped_early": r.stopped_early,
                            "stop_decision": r.stop_decision,
                            "completion_reason": r.completion_reason,
                        }
                        for r, row in zip(run["results"], rows)
                    ]
                    for name, run in runs.items()
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
