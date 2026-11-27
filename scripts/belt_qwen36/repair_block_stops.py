"""Add the probes a v1 shard missed at indented blank lines.

Shards generated before BELT_v2 stopped only on "\\n\\n", so block ends at
"\\n    \\n" were never recorded or probed. Block ends are a function of the
chain's text, so they can be recomputed exactly: the chain is replayed through
BELT_v2's generator with an engine that returns the recorded tokens
(replay_block_stops.ReplayEngine). The forced answers at the new query points
are then generated with BELT's own request builder, seed keys, engine and
grader, exactly as single_cot_probe would. Existing probes are kept as they are.

    RL_PROMPT_CONTRACT=qwen3.6-35b-a3b-fp8-native-thinking-v1 CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=<BELT_v2>/scripts/source_variants/olmo_code:<BELT_v2> \
    python repair_block_stops.py --shard-dir full/fit/shard0 --out-dir full/fit/shard0_v2 \
        --manifest pool/manifest.jsonl --model-path <snapshot>

Writes <out-dir>/chains.jsonl and repair_summary.json; the input is not touched.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay_block_stops import ReplayEngine  # noqa: E402

REVISION = "c653bba46dfa06e9658dcc0b0dc87fe4685f14a2"
CONTRACT = "qwen3.6-35b-a3b-fp8-native-thinking-v1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--dry-run", action="store_true", help="replay only, no GPU")
    args = p.parse_args()

    from rl_controller import rollout_contract as rc
    from rl_controller.scripts import single_cot_probe as scp
    from rl_controller.single_cot import run_chain_batch

    if rc.active_prompt_contract() != CONTRACT or len(rc.block_stop_strings()) < 2:
        raise SystemExit("run with RL_PROMPT_CONTRACT set and BELT_v2 on PYTHONPATH")

    run_args = scp.parse_args([
        "--manifest", args.manifest, "--split", "fit", "--split-salt", "qwen36-belt-2026-11-27-v1",
        "--model", "Qwen/Qwen3.6-35B-A3B-FP8", "--revision", REVISION,
        "--model-path", args.model_path, "--prompt-contract", CONTRACT,
        "--chains", "4", "--max-chains-per-question", "4", "--budget", "16384",
        "--block-cap", "128", "--query-stride", "4", "--max-forced-tokens", "32",
        "--max-model-len", "20480", "--gpu-mem-util", "0.92", "--seed", "0",
        "--out-dir", args.out_dir,
    ])

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    encode, decode = scp.tokenizer_codec(tok)
    manifest = {}
    for line in open(args.manifest):
        row = json.loads(line)
        manifest[row["qkey"]] = row
    chains = [json.loads(l) for l in open(Path(args.shard_dir) / "chains.jsonl")]

    t0 = time.time()
    completions, specs = {}, []
    for c in chains:
        prompt = scp.chain_prompt_ids(tok, manifest[c["qkey"]]["question"], encode=encode,
                                      decode=decode, instruction=scp.instruction_for("boxed"))
        key = (c["qkey"], c["rep"], "s0chain")
        completions[key] = {"ids": c["completion_ids"], "prompt_len": len(prompt)}
        specs.append({"prompt_ids": prompt, "seed_key": key, "qkey": c["qkey"], "rep": c["rep"]})
    played = run_chain_batch(ReplayEngine(tok, completions), specs, encode=encode,
                             budget=16384, query_stride=4, block_cap=128)
    replay_s = time.time() - t0

    items, targets = [], []
    for i, (c, a, spec) in enumerate(zip(chains, played, specs)):
        r = a["result"]
        if list(r.completion_ids) != list(c["completion_ids"]):
            raise SystemExit(f"replay changed the tokens of {c['qkey']}:{c['rep']}")
        if not set(c["block_ends"]) <= set(r.block_ends):
            raise SystemExit(f"replay lost a recorded block end of {c['qkey']}:{c['rep']}")
        old = {(int(b), int(p_)) for b, p_ in c["query_points"]}
        new_points = [(int(b), int(p_)) for b, p_ in a["query_points"]]
        if not old <= set(new_points):
            raise SystemExit(f"replay lost a recorded query point of {c['qkey']}:{c['rep']}")
        c["block_ends"] = list(r.block_ends)
        c["query_points"] = [[b, p_] for b, p_ in new_points]
        missing = [q for q in new_points if q not in old]
        if missing:
            items.append({"prompt_ids": spec["prompt_ids"], "seed_key": spec["seed_key"],
                          "completion_ids": c["completion_ids"], "query_points": missing})
            targets.append(i)

    summary = {"chains": len(chains), "chains_repaired": len(targets),
               "probes_before": sum(len(c["probes"]) for c in chains),
               "probes_added": sum(len(it["query_points"]) for it in items),
               "replay_s": round(replay_s, 1), "source": str(Path(args.shard_dir).resolve())}
    print(json.dumps(summary), flush=True)
    if args.dry_run:
        return

    t0 = time.time()
    if items:
        _, engine = scp._load_backend(run_args)
        reqs, meta = scp.build_probe_requests(items, encode=encode, max_forced_tokens=32, draws=1)
        outs = scp._generate_chunked(engine, reqs)
        added_rows = [{"probes": []} for _ in items]
        scp.attach_probes(added_rows, meta, outs, draws=1, stored_draw0=None, grade=scp.grade_forced)
        for row, i in zip(added_rows, targets):
            merged = chains[i]["probes"] + row["probes"]
            chains[i]["probes"] = sorted(merged, key=lambda pr: int(pr["block"]))
    for c in chains:
        have = [(int(pr["block"]), int(pr["prefix_len"])) for pr in c["probes"]]
        if have != sorted((int(b), int(p_)) for b, p_ in c["query_points"]):
            raise SystemExit(f"probe grid mismatch after repair: {c['qkey']}:{c['rep']}")
        scp.validate_chain(c)
    summary["probes_after"] = sum(len(c["probes"]) for c in chains)
    summary["forced_s"] = round(time.time() - t0, 1)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "chains.jsonl.partial"
    with open(tmp, "w") as f:
        for c in chains:
            f.write(json.dumps(c) + "\n")
    tmp.replace(out / "chains.jsonl")
    (out / "repair_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
