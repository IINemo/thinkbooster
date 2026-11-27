"""Replay recorded chains through BELT's generator with a fake vLLM engine.

    RL_PROMPT_CONTRACT=qwen3.6-35b-a3b-fp8-native-thinking-v1 \
    PYTHONPATH=<BELT>/scripts/source_variants/olmo_code:<BELT> \
    python replay_block_stops.py --chains chains.jsonl --manifest manifest.jsonl --tokenizer <snapshot>

The fake engine returns the chain's recorded tokens one at a time and stops the
way vLLM does: at the first token whose text completes a stop string (text cut
after the stop string, token kept), or at max_tokens. Run on an unpatched
checkout it must reproduce the recorded block ends exactly; on a patched one
the block ends should match the blank-line separators in the text.
"""

import argparse
import json
import re
from types import SimpleNamespace

BLANK = re.compile(r"\n[ \t]*\n")


def separators(text):
    out, pos = 0, 0
    for m in BLANK.finditer(text):
        if text[pos : m.start()].strip():
            out += 1
        pos = m.end()
    return out


class ReplayEngine:
    def __init__(self, tok, completions):
        self.tok, self.completions = tok, completions

    def generate_batch(self, requests):
        return [self._one(req) for req in requests]

    def _one(self, req):
        full = self.completions[req.seed_key[0]]
        start = len(req.prompt_ids) - full["prompt_len"]
        ids = full["ids"]
        taken = []
        for i in range(start, min(len(ids), start + req.max_tokens)):
            taken.append(ids[i])
            text = self.tok.decode(taken)
            hits = [text.find(s) + len(s) for s in req.stop if s in text]
            if hits:
                end = min(hits)
                return SimpleNamespace(token_ids=taken, text=text[:end], finish_reason="stop",
                                       stop_reason=None, hit_eos=False)
        text = self.tok.decode(taken)
        return SimpleNamespace(token_ids=taken, text=text, finish_reason="length",
                               stop_reason=None, hit_eos=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chains", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    from transformers import AutoTokenizer

    from rl_controller import rollout_contract as rc
    from rl_controller.scripts import single_cot_probe as scp
    from rl_controller.single_cot import run_chain_batch

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    encode, decode = scp.tokenizer_codec(tok)
    manifest = {json.loads(l)["qkey"]: json.loads(l) for l in open(args.manifest)}
    chains = [json.loads(l) for l in open(args.chains)]
    if args.limit:
        chains = chains[: args.limit]

    stops = rc.block_stop_strings() if hasattr(rc, "block_stop_strings") else [rc.BLOCK_STOP_STRING]
    print(json.dumps({"contract": rc.active_prompt_contract(), "stops": len(stops)}))

    completions, specs = {}, []
    for c in chains:
        prompt = scp.chain_prompt_ids(tok, manifest[c["qkey"]]["question"], encode=encode,
                                      decode=decode, instruction=scp.instruction_for("boxed"))
        key = (c["qkey"], c["rep"], "s0chain")
        completions[key] = {"ids": c["completion_ids"], "prompt_len": len(prompt)}
        specs.append({"prompt_ids": prompt, "seed_key": key})
    engine = ReplayEngine(tok, completions)
    played = run_chain_batch(engine, specs, encode=encode, budget=16384, query_stride=4,
                             block_cap=128)

    same_as_recorded = exact_text = superset = 0
    rows = []
    for c, a in zip(chains, played):
        r = a["result"]
        ends = list(r.block_ends)
        text = decode(c["completion_ids"])
        seps = separators(text)
        same_as_recorded += ends == list(c["block_ends"])
        superset += set(c["block_ends"]) <= set(ends)
        exact_text += abs(len(ends) - seps) <= 1
        rows.append({"chain": f"{c['qkey']}:{c['rep']}", "recorded": len(c["block_ends"]),
                     "replayed": len(ends), "text_separators": seps,
                     "query_points": len(a["query_points"]), "recorded_probes": len(c["probes"]),
                     "same_tokens": list(r.completion_ids) == list(c["completion_ids"])})
    print(json.dumps({"chains": len(chains), "block_ends_same_as_recorded": same_as_recorded,
                      "recorded_ends_subset_of_replayed": superset,
                      "replayed_matches_text_within_1": exact_text,
                      "tokens_reproduced": sum(x["same_tokens"] for x in rows)}))
    for x in rows:
        if x["recorded"] != x["replayed"] or not x["same_tokens"]:
            print(json.dumps(x))


if __name__ == "__main__":
    main()
