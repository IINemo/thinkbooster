"""Quality audit of BELT chain files for the Qwen3.6 run.

    python audit_chains.py --chains a/chains.jsonl [b/chains.jsonl ...] \
        --manifest pool/manifest.jsonl --tokenizer <snapshot> [--fallback natural_fallback.jsonl]

Prints one JSON report. It checks what the trainer silently depends on:

* identity: (qkey, rep) unique, every chain's qkey in the manifest fit split,
  gold copied from the manifest;
* blocks: block ends strictly increasing inside the completion; the number of
  blank-line separators in the decoded text agrees with the recorded block
  ends (a mismatch means the generator missed block boundaries);
* probes: each probe sits on a recorded block end at the 4th, 8th, ... block,
  its forced answer parses, Now accuracy by depth;
* chains with no probe at all (they carry no Now / Later labels);
* length-capped chains: whether their tail is a repetition loop;
* answers: natural answer parse rate, agreement with gold, and questions where
  every chain agrees on one answer that differs from gold (suspect gold).
"""

import argparse
import collections
import json
import re
import statistics

BLANK = re.compile(r"\n[ \t]*\n")


def repeat_fraction(ids, n=20):
    """Fraction of n-grams in ids that occur more than once (1.0 = pure loop)."""
    if len(ids) < 2 * n:
        return 0.0
    grams = collections.Counter(tuple(ids[i : i + n]) for i in range(len(ids) - n + 1))
    total = sum(grams.values())
    return sum(c for c in grams.values() if c > 1) / total


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chains", nargs="+", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--fallback", nargs="*", default=[])
    p.add_argument("--split", default="fit")
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    manifest = {}
    for line in open(args.manifest):
        row = json.loads(line)
        manifest[row["qkey"]] = row

    chains = []
    for path in args.chains:
        chains += [json.loads(line) for line in open(path) if line.strip()]
    fallback = {}
    for path in args.fallback:
        for line in open(path):
            r = json.loads(line)
            fallback[(r["qkey"], r["rep"])] = r

    rep = collections.Counter()
    issues = collections.Counter()
    examples = collections.defaultdict(list)
    seen = set()
    think, blocks, probes_per, block_len = [], [], [], []
    no_probe_think = []
    by_depth = collections.defaultdict(lambda: [0, 0, 0])  # n, parsed, correct
    capped_repeat, natural_repeat = [], []
    answers = collections.defaultdict(list)

    def flag(name, chain, detail=""):
        issues[name] += 1
        if len(examples[name]) < 3:
            examples[name].append(f"{chain['qkey']}:{chain['rep']} {detail}".strip())

    for c in chains:
        key = (c["qkey"], c["rep"])
        if key in seen:
            flag("duplicate_chain", c)
        seen.add(key)
        row = manifest.get(c["qkey"])
        if row is None:
            flag("qkey_not_in_manifest", c)
            continue
        if row["split"] != args.split:
            flag("wrong_split", c, row["split"])
        if int(row["gold"]) != int(c["gold"]):
            flag("gold_mismatch_vs_manifest", c)

        ids = c["completion_ids"]
        ends = [int(e) for e in c["block_ends"]]
        rep[c["outcome"]] += 1
        think.append(len(ids))
        if ends != sorted(set(ends)) or (ends and (ends[0] <= 0 or ends[-1] > len(ids))):
            flag("block_ends_not_increasing_in_range", c)
        text = tok.decode(ids)
        # Blank-line separators that end a non-empty block, as the generator counts them.
        seps, pos = 0, 0
        for m in BLANK.finditer(text):
            if text[pos : m.start()].strip():
                seps += 1
            pos = m.end()
        recorded = len(ends)
        if abs(seps - recorded) > 1:
            flag("block_count_mismatch", c, f"text {seps} vs recorded {recorded}")
        blocks.append(recorded)
        prev = 0
        for e in ends:
            block_len.append(e - prev)
            prev = e

        probes = c.get("probes") or []
        probes_per.append(len(probes))
        if not probes:
            no_probe_think.append(len(ids))
        end_set = set(ends)
        for pr in probes:
            b = int(pr["block"])
            if b % 4 != 0:
                flag("probe_not_on_4th_block", c, f"block {b}")
            if int(pr["prefix_len"]) not in end_set:
                flag("probe_not_at_block_end", c, f"prefix {pr['prefix_len']}")
            d = by_depth[min(b // 16, 7)]
            d[0] += 1
            forced = pr.get("forced")
            if forced is not None:
                d[1] += 1
                d[2] += int(int(forced) == int(c["gold"]))

        if c["outcome"] == "length":
            capped_repeat.append(repeat_fraction(ids[-3000:]))
        else:
            natural_repeat.append(repeat_fraction(ids[-3000:]))

        ans = c.get("answer")
        fb = fallback.get(key)
        if ans is None and fb is not None:
            ans = fb.get("forced")
        answers[c["qkey"]].append(None if ans is None else int(ans))

    n = len(chains)
    parsed = [a for v in answers.values() for a in v if a is not None]
    agree = sum(
        int(a == int(manifest[q]["gold"])) for q, v in answers.items() for a in v if a is not None
    )
    suspect = []
    for q, v in answers.items():
        vals = [a for a in v if a is not None]
        if len(vals) >= 2 and len(set(vals)) == 1 and vals[0] != int(manifest[q]["gold"]):
            suspect.append((q, manifest[q]["source"], vals[0], int(manifest[q]["gold"])))

    report = {
        "chains": n,
        "questions": len(answers),
        "outcome": dict(rep),
        "issues": dict(issues),
        "issue_examples": dict(examples),
        "thinking_tokens": {
            "mean": round(statistics.mean(think)),
            "median": pct(think, 0.5),
            "p90": pct(think, 0.9),
        },
        "blocks_per_chain": {"median": pct(blocks, 0.5), "p10": pct(blocks, 0.1), "p90": pct(blocks, 0.9)},
        "block_tokens": {"median": pct(block_len, 0.5), "p90": pct(block_len, 0.9), "p99": pct(block_len, 0.99), "max": max(block_len) if block_len else None},
        "probes_per_chain": {"mean": round(statistics.mean(probes_per), 2), "zero": sum(x == 0 for x in probes_per)},
        "no_probe_chains_thinking_tokens": sorted(no_probe_think)[:10],
        "now_by_depth_block16": {
            f"blocks {16 * k}-{16 * k + 15}": {
                "probes": v[0],
                "parsed": round(v[1] / v[0], 3) if v[0] else None,
                "correct": round(v[2] / v[0], 3) if v[0] else None,
            }
            for k, v in sorted(by_depth.items())
        },
        "repeat_fraction_last3k": {
            "capped_median": pct(capped_repeat, 0.5),
            "capped_over_0.5": sum(x > 0.5 for x in capped_repeat),
            "capped_n": len(capped_repeat),
            "natural_median": pct(natural_repeat, 0.5),
        },
        "answers": {
            "parsed_fraction": round(len(parsed) / n, 3) if n else None,
            "agree_with_gold_of_parsed": round(agree / len(parsed), 3) if parsed else None,
            "suspect_gold_questions": len(suspect),
            "suspect_examples": suspect[:8],
        },
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
