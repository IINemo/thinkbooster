"""Write a ThinkBooster BELT policy file from BELT's published seed-0 policy fits.

BELT's repository ships the fitted seed-0 policies of the Qwen3-1.7B
ReProbe-initialized probe in ``examples/qwen3-1.7b-sample/policies``: 17
price-refitted rows per feature interface, with their normalization. This
script picks one row and writes it in ``thinkbooster-belt-policy/v1`` format
with an explicit decision-logit threshold.

The paper's recommended configuration (H-T) uses one fitted direction and moves
its threshold. ``--lambda-index 7`` (lambda = 1.15e-5) is the grid row closest
to the paper's threshold reference price of 1e-5.

    python scripts/belt_tutorial/make_policy.py \
        --sample-policy ../BELT/examples/qwen3-1.7b-sample/policies/belt_h_r_seed0.json \
        --lambda-index 7 --threshold 0.0 --out policies/h_l7_t0.json
"""

import argparse
import json
from pathlib import Path


def build(sample, lambda_index, threshold):
    rows = sample["rows"]
    row = next(r for r in rows if r["lambda_index"] == lambda_index)
    method = sample["method"]
    if method == "BELT-H-R":
        feature_space = "H"
    elif method == "BELT-B-R":
        feature_space = "B"
    else:
        raise ValueError(f"Unsupported sample policy method {method!r}")
    policy = {
        "schema": "thinkbooster-belt-policy/v1",
        "checkpoint_sha256": sample["checkpoint"]["sha256"],
        "feature_space": feature_space,
        "weights": [float(x) for x in row["w"]],
        "intercept": float(row["b"]),
        "means": [float(x) for x in sample["mean"]],
        "stds": [float(x) for x in sample["std"]],
        "threshold": float(threshold),
    }
    if feature_space == "H":
        policy["hidden_dtype"] = sample["hidden_dtype"]
    else:
        policy["length_scale"] = float(sample["length_scale"])
        policy["logit_clip"] = float(sample["logit_clip"])
    provenance = {
        "source": "EdisonNi-hku/BELT examples/qwen3-1.7b-sample",
        "method": method,
        "lambda_index": lambda_index,
        "lambda": row["lambda"],
        "kappa": row["kappa"],
        "checkpoint": sample["checkpoint"],
        "threshold": float(threshold),
    }
    return policy, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample-policy", required=True, type=Path)
    parser.add_argument("--lambda-index", required=True, type=int)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    sample = json.loads(args.sample_policy.read_text())
    policy, provenance = build(sample, args.lambda_index, args.threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(policy))
    args.out.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )
    print(
        f"{args.out}: {policy['feature_space']} lambda={provenance['lambda']:.3g} "
        f"threshold={args.threshold}"
    )


if __name__ == "__main__":
    main()
