"""Export compact BELT tutorial data for the explainer page.

Reads never-stop runs with forced answers and writes, into --out:

* ``trajectories.json``: one record per trajectory with its decision points
  (prefix tokens, the four beliefs, the H policy logit, forced-answer
  correctness and text), plus reasoning snippets for the featured stories.
* ``frontiers.json``: accuracy against reasoning tokens for BELT-H (threshold
  sweep), BELT-B, the Now belief alone and a fixed token cap, per dataset.
* ``policies_b.json``: the 17 price-refitted BELT-B-R policies of seed 0.

    python scripts/belt_tutorial/export_artifact.py --root <root> --out <dir> \
        --story math500:289 --story math500:285 ...
"""

import argparse
import copy
import json
import math
from pathlib import Path

import analyze as A

LOGIT_CLIP = 1e-4


def b_logit(row, sample, d):
    x = []
    for k in ("q", "v", "r", "stability"):
        p = min(max(d["beliefs"][k], LOGIT_CLIP), 1 - LOGIT_CLIP)
        x.append(math.log(p / (1 - p)))
    x.append(d["prefix_tokens"] / sample["length_scale"])
    return row["b"] + sum(
        w * (xi - m) / s for w, xi, m, s in zip(row["w"], x, sample["mean"], sample["std"])
    )


def rescored(samples, fn):
    out = copy.deepcopy(samples)
    for s in out:
        for d in A.decisions(s):
            d["policy_logit"] = fn(d)
    return out


def curve(samples, forced, max_points=140):
    points = A.frontier(samples, forced)
    points.sort(key=lambda p: p["token_saving"])
    # Keep every Pareto point (no other point has more saving and at least the
    # same accuracy), so readings from the page are exact, plus an even sample
    # of the rest for the curve's shape.
    pareto, best = set(), -1.0
    for k in range(len(points) - 1, -1, -1):
        if points[k]["accuracy"] > best + 1e-12:
            pareto.add(k)
            best = points[k]["accuracy"]
    step = max(1, len(points) // max_points)
    keep = [p for k, p in enumerate(points) if k in pareto or k % step == 0 or k == len(points) - 1]
    return [
        [round(p["token_saving"], 4), round(p["accuracy"], 4), round(p["mean_tokens"], 1),
         None if math.isinf(p["threshold"]) else round(p["threshold"], 3), round(p["stop_rate"], 3)]
        for p in keep
    ]


def snippet(steps, upto, width=320):
    text = "".join((s.get("raw_text") or s.get("text") or "") for s in steps[:upto])
    text = text.replace("<think>\n", "").strip()
    return text[-width:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", default=["math500", "minerva"])
    parser.add_argument("--story", action="append", default=[], help="dataset:index")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    stories = {(s.split(":")[0], int(s.split(":")[1])) for s in args.story}
    sample_b = json.loads((args.root / "policies/sample/belt_b_r_seed0.json").read_text())
    row7 = next(r for r in sample_b["rows"] if r["lambda_index"] == 7)

    records, frontiers = [], {}
    for ds in args.datasets:
        run = args.root / "runs" / ds / "natural" / "seed1"
        samples = A.load_run(run)
        forced = A.load_forced(run / "forced.jsonl")
        for s in samples:
            rows = forced.get(s["index"], {})
            pts = []
            for k, d in enumerate(A.decisions(s)):
                b = d["beliefs"]
                f = rows.get(k)
                pts.append(
                    [d["prefix_tokens"], round(b["q"], 4), round(b["v"], 4), round(b["r"], 4),
                     round(b["stability"], 4), round(d["policy_logit"], 3),
                     -1 if f is None else int(f["correct"]),
                     "" if f is None else f["forced_answer"][:28]]
                )
            rec = {
                "ds": ds,
                "i": s["index"],
                "q": s["question"][:600],
                "gold": str(s["gold_answer"])[:60],
                "ok": int(A.is_correct(s)),
                "ans": str(s.get("extracted_answer") or "")[:60],
                "T": A.thinking_tokens(s),
                "why": s.get("completion_reason"),
                "p": pts,
            }
            if (ds, s["index"]) in stories:
                rec["x"] = [snippet(s["steps"], d["step"]) for d in A.decisions(s)]
            records.append(rec)
        frontiers[ds] = {
            "BELT-H": curve(samples, forced),
            "BELT-B": curve(rescored(samples, lambda d: b_logit(row7, sample_b, d)), forced),
            "Now only": curve(
                rescored(samples, lambda d: math.log(max(d["beliefs"]["q"], 1e-6) / max(1 - d["beliefs"]["q"], 1e-6))),
                forced,
            ),
            "Fixed cap": curve(rescored(samples, lambda d: d["prefix_tokens"]), forced),
        }

    (args.out / "trajectories.json").write_text(
        "[\n" + ",\n".join(json.dumps(r, separators=(",", ":")) for r in records) + "\n]\n"
    )
    (args.out / "frontiers.json").write_text(json.dumps(frontiers, separators=(",", ":")))
    (args.out / "policies_b.json").write_text(
        json.dumps(
            {
                "features": ["Now", "End", "Later", "Same", "Length"],
                "mean": sample_b["mean"],
                "std": sample_b["std"],
                "length_scale": sample_b["length_scale"],
                "rows": [{"lambda": r["lambda"], "w": r["w"], "b": r["b"]} for r in sample_b["rows"]],
            },
            separators=(",", ":"),
        )
    )
    print(f"{len(records)} trajectories, stories {sorted(stories)} -> {args.out}")


if __name__ == "__main__":
    main()
