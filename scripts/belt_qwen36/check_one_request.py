"""Check edit 9 (one request per chain) on recorded chains, on CPU.

    RL_PROMPT_CONTRACT=qwen3.6-35b-a3b-fp8-native-thinking-v1 \
    PYTHONPATH=<BELT>/scripts/source_variants/olmo_code:<BELT> \
    python check_one_request.py --chains chains.jsonl --manifest manifest.jsonl --tokenizer <snapshot>

For each recorded chain, two runs over the same tokens:

- reference: BELT's block-by-block loop, served from the full recorded tokens;
- one request: the chain is first cut where the single request would stop
  (thinking closed, then a block stop), then run_chain_batch_one_request
  replays it.

They must agree on tokens kept, block ends, query points, outcome and answer,
with no chain falling back to the block-by-block loop.
"""

import argparse
import json

from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chains", required=True, nargs="+")
    p.add_argument("--manifest", help="question texts; without it a placeholder prompt is used, "
                   "which is enough because the replay reads only the prompt's length")
    p.add_argument("--tokenizer", required=True)
    args = p.parse_args()

    from transformers import AutoTokenizer

    from rl_controller.rollout_contract import sampling_config
    from rl_controller.scripts import single_cot_probe as scp
    from rl_controller.single_cot import _RecordedChainEngine, run_chain_batch

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    encode, decode = scp.tokenizer_codec(tok)
    eos = sampling_config()["eos_token_id"]
    eos = [eos] if isinstance(eos, int) else list(eos)
    manifest = ({json.loads(l)["qkey"]: json.loads(l) for l in open(args.manifest)}
                if args.manifest else None)
    chains = [json.loads(l) for path in args.chains for l in open(path)]

    specs, recorded = [], {}
    for c in chains:
        if manifest is None:
            prompt = [0] * 200
        else:
            prompt = scp.chain_prompt_ids(tok, manifest[c["qkey"]]["question"], encode=encode,
                                          decode=decode, instruction=scp.instruction_for("boxed"))
        key = (c["qkey"], c["rep"], "s0chain")
        recorded[key] = c["completion_ids"]
        specs.append({"prompt_ids": prompt, "seed_key": key})

    class OneRequestFake:
        """Stands in for vLLM on the single request: the recorded tokens, cut by
        the single request's stop strings exactly as vLLM would cut them."""

        def generate_batch(self, requests):
            return [self._one(r) for r in requests]

        def _one(self, r):
            # Same rule as _RecordedChainEngine, but a whole chain is up to 16k
            # tokens, so look for the stop in a short trailing window first
            # (every stop string is under 30 characters) and decode in full
            # only to cut the text.
            ids = recorded[r.seed_key[0]][: r.max_tokens]
            for i, t in enumerate(ids):
                if t in eos and i == len(ids) - 1:
                    return SimpleNamespace(token_ids=ids, text=decode(ids[:-1]),
                                           finish_reason="stop", stop_reason=t, hit_eos=True)
                if any(s in decode(ids[max(0, i - 31): i + 1]) for s in r.stop):
                    text = decode(ids[: i + 1])
                    hits = [text.find(s) + len(s) for s in r.stop if s in text]
                    if hits:
                        return SimpleNamespace(token_ids=ids[: i + 1], text=text[: min(hits)],
                                               finish_reason="stop", stop_reason=None,
                                               hit_eos=False)
            return SimpleNamespace(token_ids=ids, text=decode(ids), finish_reason="length",
                                   stop_reason=None, hit_eos=False)

    kw = dict(encode=encode, budget=16384, query_stride=4, block_cap=128)
    one = run_chain_batch(OneRequestFake(), specs, one_request=True, decode=decode, **kw)

    same = {"tokens": 0, "block_ends": 0, "query_points": 0, "outcome": 0, "answer": 0}
    for spec, a in zip(specs, one):
        ref = run_chain_batch(_RecordedChainEngine(decode, len(spec["prompt_ids"]),
                                                   recorded[spec["seed_key"]], eos), [spec], **kw)[0]
        r, q = ref["result"], a["result"]
        checks = {"tokens": list(r.completion_ids) == list(q.completion_ids),
                  "block_ends": list(r.block_ends) == list(q.block_ends),
                  "query_points": list(ref["query_points"]) == list(a["query_points"]),
                  "outcome": r.outcome == q.outcome, "answer": r.answer == q.answer}
        for k, v in checks.items():
            same[k] += v
        if not all(checks.values()):
            print(json.dumps({"chain": list(spec["seed_key"][:2]), **checks,
                              "ref_tokens": len(r.completion_ids),
                              "one_tokens": len(q.completion_ids)}))
    print(json.dumps({"chains": len(specs), "identical": same}))


if __name__ == "__main__":
    main()
