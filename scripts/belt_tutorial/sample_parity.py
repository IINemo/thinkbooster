"""Check ThinkBooster's BELT reader against the paper's archived scores.

BELT's repository ships ``examples/qwen3-1.7b-sample``: 16 Qwen3-1.7B
trajectories with their exact prompt and completion token IDs, the probe
points, and the beliefs, 768-d representations and policy logits that the
paper's pipeline computed there. This script feeds the same token IDs through
ThinkBooster's ``BeltRuntime.score_token_ids`` (default HF backend) and reports
the differences. It answers whether the integration reproduces the paper's
probe outputs on the paper's own inputs.

    python scripts/belt_tutorial/sample_parity.py \
        --sample ../BELT/examples/qwen3-1.7b-sample --out sample_parity.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument(
        "--policy", type=Path, help="thinkbooster policy JSON (H, lambda index 7)"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from thinkbooster.belt import BeltStoppingPolicy
    from thinkbooster.scorers.step_scorer_belt import StepScorerBELT

    chains = [json.loads(line) for line in (args.sample / "chains.jsonl").open()]
    refs = {
        (r["qkey"], r["rep"]): r
        for r in map(
            json.loads, (args.sample / "reference" / "policy_reference.jsonl").open()
        )
    }
    blob = np.fromfile(args.sample / "belt_h.f16", dtype="<f2")
    policy = BeltStoppingPolicy.from_file(args.policy) if args.policy else None
    runtime = StepScorerBELT(device=args.device).runtime

    rows = []
    for chain in chains:
        h_ref = (
            blob[chain["h"]["offset"] // 2 :][: chain["h"]["n_probes"] * 768]
            .reshape(-1, 768)
            .astype(np.float64)
        )
        ref = refs[(chain["qkey"], chain["rep"])]
        z_ref = next(p for p in ref["policies"]["BELT-H-R"] if p["lambda_index"] == 7)[
            "logit"
        ]
        for i, (probe, scores) in enumerate(zip(chain["probes"], chain["scores"])):
            prefix = probe["prefix_len"]
            ends = [e for e in chain["block_ends"] if e <= prefix]
            signal = runtime.score_token_ids(
                chain["prompt_ids"], chain["completion_ids"][:prefix], ends
            )
            hidden = np.asarray(signal.hidden, dtype=np.float64)
            row = {
                "qkey": chain["qkey"],
                "rep": chain["rep"],
                "probe": i,
                "prefix_len": prefix,
                "belief_abs_err": {
                    k: abs(signal.beliefs[k] - scores[k])
                    for k in ("q", "v", "r", "stability")
                },
                "hidden_rel_l2": float(
                    np.linalg.norm(hidden - h_ref[i])
                    / max(np.linalg.norm(h_ref[i]), 1e-12)
                ),
            }
            if policy is not None:
                row["policy_logit"] = policy.decision_logit(signal)
                row["policy_logit_ref"] = z_ref[i]
            rows.append(row)

    belief = [max(r["belief_abs_err"].values()) for r in rows]
    rel = [r["hidden_rel_l2"] for r in rows]
    report = {
        "probe_points": len(rows),
        "max_belief_abs_err": {
            "median": float(np.median(belief)),
            "max": float(max(belief)),
        },
        "hidden_rel_l2": {"median": float(np.median(rel)), "max": float(max(rel))},
    }
    if policy is not None:
        dz = [abs(r["policy_logit"] - r["policy_logit_ref"]) for r in rows]
        same = sum(
            (r["policy_logit"] >= 0) == (r["policy_logit_ref"] >= 0) for r in rows
        )
        report["policy_logit_abs_err"] = {
            "median": float(np.median(dz)),
            "max": float(max(dz)),
        }
        report["same_sign_at_threshold_0"] = same / len(rows)
    args.out.write_text(json.dumps({"summary": report, "rows": rows}, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
