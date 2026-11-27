"""Read BELT tutorial runs, calibrate thresholds and summarize results.

Runs live in ``<root>/runs/<dataset>/<policy-tag>/seed<k>/results.json``.

    # Label-free threshold calibration on a never-stop run
    python scripts/belt_tutorial/analyze.py calibrate \
        --run <root>/runs/minerva/natural/seed1 --savings 0.3 0.45 0.6

    # Accuracy and generated tokens of every finished run
    python scripts/belt_tutorial/analyze.py summarize --root <root> --out summary.json
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

CODE_EVALUATORS = ("human_eval_plus", "mbpp_plus")


def is_correct(sample):
    evals = sample.get("eval") or {}
    for name in CODE_EVALUATORS:
        if name in evals:
            return bool(evals[name].get("is_correct"))
    if "exact_match" in evals:
        return bool(evals["exact_match"].get("is_correct"))
    return bool(sample.get("is_correct"))


def output_tokens(sample):
    return int((sample.get("token_stats") or {}).get("output_tokens", 0))


def decisions(sample):
    return (sample.get("metadata") or {}).get("decisions") or []


def load_run(run_dir):
    path = Path(run_dir) / "results.json"
    return json.loads(path.read_text())


def projected_cost(sample, threshold):
    """Generated tokens if the run had stopped at the first logit >= threshold.

    Uses only the never-stop trace: the reasoning prefix at the stop plus the
    run's own answer length. No correctness labels are involved.
    """
    natural = output_tokens(sample)
    answer = answer_tokens(sample)
    for d in decisions(sample):
        if d["policy_logit"] >= threshold:
            return min(natural, d["prefix_tokens"] + answer)
    return natural


def answer_tokens(sample):
    """Approximate tokens of the final answer: total minus the last decision prefix."""
    steps = sample.get("steps") or []
    text = sample.get("answer_step") or ""
    if not text and steps:
        text = steps[-1] if isinstance(steps[-1], str) else ""
    # Roughly 3.5 characters per token for Qwen3 on English/LaTeX answers.
    return max(1, int(len(text) / 3.5))


def projected_saving(samples, threshold):
    natural = sum(output_tokens(s) for s in samples)
    cost = sum(projected_cost(s, threshold) for s in samples)
    return 1.0 - cost / natural


def calibrate(samples, targets):
    logits = sorted({d["policy_logit"] for s in samples for d in decisions(s)})
    if not logits:
        raise ValueError("Run has no BELT decisions")
    out = []
    for target in targets:
        # Saving falls as the threshold rises; take the highest threshold whose
        # projected saving still reaches the target.
        lo, hi = 0, len(logits) - 1
        best = logits[0]
        while lo <= hi:
            mid = (lo + hi) // 2
            if projected_saving(samples, logits[mid]) >= target:
                best = logits[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        out.append(
            {
                "target_saving": target,
                "threshold": best,
                "projected_saving": projected_saving(samples, best),
                "projected_stop_rate": sum(
                    any(d["policy_logit"] >= best for d in decisions(s))
                    for s in samples
                )
                / len(samples),
            }
        )
    return out


def summarize_run(samples):
    n = len(samples)
    correct = sum(is_correct(s) for s in samples)
    tokens = [output_tokens(s) for s in samples]
    reasons = defaultdict(int)
    for s in samples:
        reasons[s.get("completion_reason") or "unknown"] += 1
    return {
        "n": n,
        "accuracy": correct / n if n else math.nan,
        "correct": correct,
        "mean_output_tokens": sum(tokens) / n if n else math.nan,
        "total_output_tokens": sum(tokens),
        "belt_stops": sum(
            1 for s in samples if any(d.get("stop") for d in decisions(s))
        ),
        "completion_reasons": dict(reasons),
    }


def summarize(root):
    rows = []
    for results in sorted(Path(root).glob("runs/*/*/seed*/results.json")):
        seed_dir = results.parent
        tag_dir = seed_dir.parent
        dataset = tag_dir.parent.name
        samples = json.loads(results.read_text())
        row = {
            "dataset": dataset,
            "policy": tag_dir.name,
            "seed": seed_dir.name,
            **summarize_run(samples),
        }
        rows.append(row)
    natural = {
        (r["dataset"], r["seed"]): r for r in rows if r["policy"] == "natural"
    }
    for r in rows:
        base = natural.get((r["dataset"], r["seed"]))
        if base:
            r["token_saving_vs_natural"] = (
                1 - r["total_output_tokens"] / base["total_output_tokens"]
            )
            r["accuracy_delta_vs_natural"] = r["accuracy"] - base["accuracy"]
    return rows


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--run", required=True)
    cal.add_argument("--savings", nargs="+", type=float, required=True)
    cal.add_argument("--out")
    summ = sub.add_parser("summarize")
    summ.add_argument("--root", required=True)
    summ.add_argument("--out")
    args = parser.parse_args()

    if args.command == "calibrate":
        result = calibrate(load_run(args.run), args.savings)
    else:
        result = summarize(args.root)
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
