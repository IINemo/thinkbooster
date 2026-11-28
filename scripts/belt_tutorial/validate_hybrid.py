"""Held-out check of "stop on the BELT-H score or a token cap".

On one sample per problem, the combined rule looked better than either part,
but its threshold and cap were picked on the same runs it was scored on. This
script picks every method's settings on one half of the problems and scores
them on the other half, over many random splits, so the comparison is fair:

* ``BELT-H``: one setting, the stop threshold.
* ``cap``: one setting, the token cap (stop at the first decision point past it).
* ``BELT-H(0) + cap``: the fitted threshold 0 kept fixed, one setting, the cap.
* ``BELT-H + cap``: two settings, threshold and cap.

For a target saving, each method takes the setting with the best training
accuracy among those that save at least the target on the training half. The
held-out half reports accuracy and the saving actually reached. A transfer
check fits on all of one dataset and scores another.

It replays the page data (``trajectories.json``): a stop takes the graded
forced answer at that decision point and costs its prefix plus the forced
answer; a run with no stop keeps its own answer and length.

    python scripts/belt_tutorial/validate_hybrid.py \
        --data reports/belt_tutorial/trajectories.json --out hybrid_validation.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

THRESHOLDS = np.concatenate([np.arange(-10.0, 6.01, 0.5), [np.inf]])
CAPS = np.concatenate([np.arange(1000, 12001, 100), [np.inf]])
TARGETS = (0.2, 0.3, 0.4)


def outcome_tables(trajs):
    """Per run and setting pair: correct (0/1) and token cost.

    Returns arrays of shape (runs, len(THRESHOLDS), len(CAPS)) and each run's
    natural length.
    """
    n, nt, nc = len(trajs), len(THRESHOLDS), len(CAPS)
    correct = np.zeros((n, nt, nc))
    cost = np.zeros((n, nt, nc))
    natural = np.zeros(n)
    for r, t in enumerate(trajs):
        natural[r] = t["T"]
        # Only graded decision points can be stopped at (as on the page).
        pts = [p for p in t["p"] if p[6] != -1]
        big = len(pts)
        tok = np.array([p[0] for p in pts], dtype=float)
        score = np.array([p[5] for p in pts], dtype=float)
        first_thr = np.array(
            [next((k for k in range(big) if score[k] >= th), big) for th in THRESHOLDS]
        )
        first_cap = np.array(
            [next((k for k in range(big) if tok[k] >= c), big) for c in CAPS]
        )
        k = np.minimum(first_thr[:, None], first_cap[None, :])
        stop_ok = np.array([p[6] for p in pts] + [t["ok"]], dtype=float)
        stop_cost = np.array([p[0] + (p[8] or 0) for p in pts] + [t["T"]], dtype=float)
        correct[r] = stop_ok[k]
        cost[r] = stop_cost[k]
    return correct, cost, natural


# Each method is a set of allowed (threshold index, cap index) pairs.
def method_masks():
    nt, nc = len(THRESHOLDS), len(CAPS)
    inf_t, inf_c = nt - 1, nc - 1
    zero_t = int(np.argmin(np.abs(THRESHOLDS - 0.0)))
    masks = {}
    m = np.zeros((nt, nc), bool)
    m[:, inf_c] = True
    masks["BELT-H"] = m
    m = np.zeros((nt, nc), bool)
    m[inf_t, :] = True
    masks["cap"] = m
    m = np.zeros((nt, nc), bool)
    m[zero_t, :] = True
    masks["BELT-H(0) + cap"] = m
    masks["BELT-H + cap"] = np.ones((nt, nc), bool)
    return masks


def evaluate(correct, cost, natural, idx):
    acc = correct[idx].mean(0)
    saving = 1 - cost[idx].sum(0) / natural[idx].sum()
    return acc, saving


def pick(acc, saving, mask, target):
    ok = mask & (saving >= target - 1e-12)
    if not ok.any():
        return None
    # Best accuracy, then most saving.
    key = np.where(ok, acc * 1e6 + saving, -np.inf)
    return np.unravel_index(int(np.argmax(key)), key.shape)


def setting(ij):
    t, c = THRESHOLDS[ij[0]], CAPS[ij[1]]
    return {
        "threshold": None if np.isinf(t) else float(t),
        "cap": None if np.isinf(c) else float(c),
    }


def split_study(tables, splits, seed):
    correct, cost, natural = tables
    n = len(natural)
    rng = np.random.default_rng(seed)
    masks = method_masks()
    rows = {m: {tg: [] for tg in TARGETS} for m in masks}
    for _ in range(splits):
        perm = rng.permutation(n)
        train, test = perm[: n // 2], perm[n // 2 :]
        acc_tr, sav_tr = evaluate(correct, cost, natural, train)
        acc_te, sav_te = evaluate(correct, cost, natural, test)
        for m, mask in masks.items():
            for tg in TARGETS:
                ij = pick(acc_tr, sav_tr, mask, tg)
                if ij is not None:
                    rows[m][tg].append((acc_te[ij], sav_te[ij]))
    out = {}
    for tg in TARGETS:
        res = {}
        for m in masks:
            a = np.array(rows[m][tg])
            res[m] = {
                "test_accuracy": float(a[:, 0].mean()),
                "test_saving": float(a[:, 1].mean()),
                "splits": len(a),
            }
        # Paired differences on the same splits, against the combined rules.
        for combo in ("BELT-H + cap", "BELT-H(0) + cap"):
            base = np.array(rows[combo][tg])
            for other in ("BELT-H", "cap"):
                o = np.array(rows[other][tg])
                d = base[:, 0] - o[:, 0]
                res[f"{combo} minus {other}"] = {
                    "mean_points": float(100 * d.mean()),
                    "p05_points": float(100 * np.percentile(d, 5)),
                    "p95_points": float(100 * np.percentile(d, 95)),
                    "share_ahead": float((d > 0).mean()),
                    "share_tied": float((d == 0).mean()),
                }
        out[str(tg)] = res
    return out


def transfer(src, dst):
    acc_s, sav_s = evaluate(*src, np.arange(len(src[2])))
    acc_d, sav_d = evaluate(*dst, np.arange(len(dst[2])))
    out = {}
    for tg in TARGETS:
        res = {}
        for m, mask in method_masks().items():
            ij = pick(acc_s, sav_s, mask, tg)
            res[m] = {
                **setting(ij),
                "source_accuracy": float(acc_s[ij]),
                "source_saving": float(sav_s[ij]),
                "target_accuracy": float(acc_d[ij]),
                "target_saving": float(sav_d[ij]),
            }
        out[str(tg)] = res
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--splits", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    trajs = json.loads(args.data.read_text())
    tables = {
        ds: outcome_tables([t for t in trajs if t["ds"] == ds])
        for ds in sorted({t["ds"] for t in trajs})
    }
    report = {
        "splits": args.splits,
        "seed": args.seed,
        "held_out": {
            ds: split_study(tab, args.splits, args.seed) for ds, tab in tables.items()
        },
    }
    if {"math500", "minerva"} <= tables.keys():
        report["transfer"] = {
            "math500->minerva": transfer(tables["math500"], tables["minerva"]),
            "minerva->math500": transfer(tables["minerva"], tables["math500"]),
        }
    args.out.write_text(json.dumps(report, indent=2))
    for ds, res in report["held_out"].items():
        print(f"== {ds}: held-out halves, {args.splits} splits")
        for tg, r in res.items():
            line = "  ".join(
                f"{m} {100 * r[m]['test_accuracy']:.1f}%/{100 * r[m]['test_saving']:.0f}%"
                for m in method_masks()
            )
            print(f"  target {tg}: {line}")
            for k, v in r.items():
                if "minus" in k:
                    print(
                        f"    {k}: {v['mean_points']:+.2f} pt "
                        f"[{v['p05_points']:+.1f}, {v['p95_points']:+.1f}], "
                        f"ahead {100 * v['share_ahead']:.0f}%"
                    )
    for name, res in report.get("transfer", {}).items():
        print(f"== transfer {name}")
        for tg, r in res.items():
            print(
                f"  target {tg}: "
                + "  ".join(
                    f"{m} {100 * v['target_accuracy']:.1f}%/{100 * v['target_saving']:.0f}%"
                    for m, v in r.items()
                )
            )


if __name__ == "__main__":
    main()
