"""Add the Qwen3.6-35B-A3B contract to a BELT checkout, without changing BELT's paths.

BELT (github.com/EdisonNi-hku/BELT, pinned at ``BELT_COMMIT``) ships its Qwen3
and OLMo-3 recipes as code snapshots. Qwen3.6 has no ReProbe head, so it
follows the OLMo path: ``scripts/source_variants/olmo_code`` generates and
labels chains, ``scripts/source_variants/olmo_belt`` trains the self-contained
reader, and the root package builds the question pool and runs the
natural-answer fallback. This script makes these edits, each asserted
against the exact text it replaces, so a moved line fails loudly:

1. ``rl_controller/qwen36_contract.py`` is copied into all three packages, and
   each ``rollout_contract.py`` gains a block that registers the contract and
   wraps ``contract_identity``, ``build_prompt_ids`` and ``sampling_config``.
   Other contracts pass through to BELT's own code unchanged.
2. Every ``single_cot_probe._load_backend`` turns off image and video inputs
   for this contract (the checkpoint is a vision-language model; we send text).
3. The natural-answer fallback accepts ``--prompt-contract qwen36``.
4. The pool builder screens Minerva Math as an evaluation set too.
5. The ``olmo_belt`` trainer loads this backbone text-only, truncated to the
   deepest feature layer, with the FP8 weights dequantized to bf16. Layers
   after the feature layer are never read, so truncation does not change the
   features. ``QWEN36_BACKBONE_DEVICE`` can put the backbone on a second GPU.
6. For this contract the reader's causal encoder turns off PyTorch's
   MultiheadAttention fast path and drops an all-False key-padding mask; both
   would otherwise build a [heads, T, T] mask, 8 GiB per 16k-token chain.
7. Generation stops at indented blank lines too ("\\n" + spaces + "\\n"), which
   BELT's block separator already counts but its "\\n\\n" stop string missed;
   Qwen3.6 writes about 8% of its paragraph breaks that way.
8. ``run_chain_batch`` can cap the chains in flight (``BELT_MAX_ACTIVE_CHAINS``)
   and log progress (``BELT_LOG_EVERY_CHAINS``). Uncapped, a 1,074-chain shard
   holds about 5M tokens of context against a 300k-token KV cache, so every
   round re-reads every chain from scratch. Unset, BELT's schedule is unchanged.
9. With ``BELT_ONE_REQUEST_CHAINS=1``, ``run_chain_batch`` samples each chain in
   one vLLM request and replays its tokens through ``draw_steps``, so block ends,
   query points and results are what the block-by-block loop records for those
   tokens. The always-continue policy never acts at a block end, so BELT's
   per-block requests only re-read the context: on Qwen3.6 (hybrid attention,
   1,056-token cache blocks) that kept the GPU busy re-reading, at about 175
   tokens/s per RTX 5880. Unset, BELT's loop is unchanged.

    python scripts/belt_qwen36/apply_belt_patch.py /path/to/BELT
    python scripts/belt_qwen36/apply_belt_patch.py --add-one-request /path/to/BELT

It refuses a checkout that is not at ``BELT_COMMIT`` or is already patched;
``--add-one-request`` adds only edit 9 to a checkout that has edits 1 to 8.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

BELT_COMMIT = "700948b5f93ad1f40ab46589c5d4399c21275e1a"
HERE = Path(__file__).resolve().parent
PACKAGES = (
    "rl_controller",
    "scripts/source_variants/olmo_code/rl_controller",
    "scripts/source_variants/olmo_belt/rl_controller",
)
MARK = "# --- Qwen3.6 contract (thinkbooster scripts/belt_qwen36/apply_belt_patch.py)"

CONTRACT_BLOCK = MARK + """
QWEN36_CONTRACT = "qwen3.6-35b-a3b-fp8-native-thinking-v1"
PROMPT_CONTRACTS = PROMPT_CONTRACTS + (QWEN36_CONTRACT,)


def _qwen36():
    from rl_controller import qwen36_contract as q
    return q


_belt_contract_identity = contract_identity
_belt_build_prompt_ids = build_prompt_ids
_belt_sampling_config = sampling_config


def contract_identity(contract: str | None = None) -> dict:
    if (contract or active_prompt_contract()) == QWEN36_CONTRACT:
        return _qwen36().identity()
    return _belt_contract_identity(contract)


def build_prompt_ids(tokenizer, question: str, *, contract: str | None = None) -> list[int]:
    if (contract or active_prompt_contract()) == QWEN36_CONTRACT:
        q = _qwen36()
        _, prompt_ids = q.build_request_payload(tokenizer, q.MODEL, question, seed=0)
        return list(prompt_ids)
    return _belt_build_prompt_ids(tokenizer, question, contract=contract)


def sampling_config(*, contract: str | None = None) -> dict:
    if (contract or active_prompt_contract()) == QWEN36_CONTRACT:
        q = _qwen36()
        cfg = q.resolved_generation_config()
        q.assert_official_generation_config(cfg)
        return {
            "temperature": cfg["temperature"],
            "top_p": cfg["top_p"],
            "top_k": cfg["top_k"],
            "min_p": cfg["min_p"],
            "repetition_penalty": cfg["repetition_penalty"],
            "max_tokens": cfg["max_tokens"],
            "eos_token_id": list(cfg["eos_token_id"]),
        }
    return _belt_sampling_config(contract=contract)


# Qwen3.6 often separates paragraphs with an indented blank line ("\\n    \\n").
# BLOCK_SEPARATOR already counts it as a block end, but the vLLM stop string
# "\\n\\n" never stops on it, so those block ends (and their probes) were lost.
_QWEN36_BLOCK_STOPS = ["\\n\\n"] + ["\\n" + " " * k + "\\n" for k in range(1, 17)] + ["\\n\\t\\n"]


def block_stop_strings() -> list:
    if active_prompt_contract() == QWEN36_CONTRACT:
        return list(_QWEN36_BLOCK_STOPS)
    return [BLOCK_STOP_STRING]
"""

RUNNER_EDITS = (
    (
        "from rl_controller.budget_episode import ABANDON, ANSWER, CONTINUE, INNER_ACTIONS\n",
        "from rl_controller.budget_episode import ABANDON, ANSWER, CONTINUE, INNER_ACTIONS\n"
        "import rl_controller.rollout_contract as _rc\n",
    ),
    (
        "            stop=[BLOCK_STOP_STRING], include_stop_str_in_output=True,",
        "            stop=_rc.block_stop_strings(), include_stop_str_in_output=True,",
    ),
)
BATCH_EDITS = (
    (
        "from rl_controller.rollout_contract import block_count, classify_stop, BLOCK_STOP_STRING\n",
        "from rl_controller.rollout_contract import block_count, classify_stop, block_stop_strings\n",
    ),
    ("req.stop != [BLOCK_STOP_STRING]", "req.stop != block_stop_strings()"),
)

CHAIN_BATCH_OLD = """                    block_cap: int = 0):
    active = []
    for spec in specs:
        records: list = []

        def observe(block, completion_ids, block_ends, _r=records):
            _r.append((int(block), len(completion_ids)))
            return {}

        gen = draw_steps(prompt_ids=spec["prompt_ids"], policy=_always_continue,
                         observe=observe, encode=encode,
                         max_blocks=10 ** 9, max_total_tokens=budget,
                         budget_left=budget, query_stride=query_stride,
                         block_cap=block_cap, request_seed_base=spec["seed_key"])
        item = {"spec": spec, "gen": gen, "query_points": records,
                "pending": gen.send(None), "result": None}
        active.append(item)

    while any(a["result"] is None for a in active):
        waiting = [a for a in active if a["result"] is None]
        outs = engine.generate_batch([a["pending"] for a in waiting])
        if len(outs) != len(waiting):
            raise RuntimeError(f"engine returned {len(outs)} outputs for {len(waiting)} requests")
        for a, out in zip(waiting, outs):
            try:
                a["pending"] = a["gen"].send(out)
            except StopIteration as stop:
                a["result"] = stop.value
                a["pending"] = None
    return active
"""
CHAIN_BATCH_NEW = """                    block_cap: int = 0, max_active: int = 0, log_every: int = 0):
    # max_active > 0 keeps at most that many chains in flight and starts the
    # next one as each finishes. Every request carries its own seed, so this
    # changes the schedule, not the chains; it keeps the in-flight contexts
    # inside vLLM's prefix cache instead of re-reading each one every round.
    import json as _json
    import time as _time

    def start(spec):
        records: list = []

        def observe(block, completion_ids, block_ends, _r=records):
            _r.append((int(block), len(completion_ids)))
            return {}

        gen = draw_steps(prompt_ids=spec["prompt_ids"], policy=_always_continue,
                         observe=observe, encode=encode,
                         max_blocks=10 ** 9, max_total_tokens=budget,
                         budget_left=budget, query_stride=query_stride,
                         block_cap=block_cap, request_seed_base=spec["seed_key"])
        return {"spec": spec, "gen": gen, "query_points": records,
                "pending": gen.send(None), "result": None}

    limit = max_active if max_active > 0 else len(specs)
    active = []
    live = []
    finished = 0
    t0 = _time.time()
    while len(active) < len(specs) and len(live) < limit:
        active.append(start(specs[len(active)]))
        live.append(active[-1])
    while live:
        outs = engine.generate_batch([a["pending"] for a in live])
        if len(outs) != len(live):
            raise RuntimeError(f"engine returned {len(outs)} outputs for {len(live)} requests")
        still = []
        for a, out in zip(live, outs):
            try:
                a["pending"] = a["gen"].send(out)
                still.append(a)
            except StopIteration as stop:
                a["result"] = stop.value
                a["pending"] = None
                finished += 1
                if log_every and (finished % log_every == 0 or finished == len(specs)):
                    print(_json.dumps({"chains_finished": finished, "of": len(specs),
                                       "elapsed_s": round(_time.time() - t0, 1)}), flush=True)
        while len(active) < len(specs) and len(still) < limit:
            active.append(start(specs[len(active)]))
            still.append(active[-1])
        live = still
    return active
"""
PROBE_CALL_OLD = """        played = run_chain_batch(engine, specs, encode=encode, budget=args.budget,
                                 query_stride=args.query_stride, block_cap=args.block_cap)
"""
PROBE_CALL_NEW = """        import os as _os
        played = run_chain_batch(engine, specs, encode=encode, budget=args.budget,
                                 query_stride=args.query_stride, block_cap=args.block_cap,
                                 max_active=int(_os.environ.get("BELT_MAX_ACTIVE_CHAINS") or 0),
                                 log_every=int(_os.environ.get("BELT_LOG_EVERY_CHAINS") or 0))
"""

ONE_REQUEST_MARK = "# --- One request per chain (thinkbooster apply_belt_patch.py edit 9)"
ONE_REQUEST_SIG_OLD = """                    block_cap: int = 0, max_active: int = 0, log_every: int = 0):
"""
ONE_REQUEST_SIG_NEW = """                    block_cap: int = 0, max_active: int = 0, log_every: int = 0,
                    one_request: bool = False, decode=None):
"""
ONE_REQUEST_DISPATCH_OLD = """    limit = max_active if max_active > 0 else len(specs)
"""
ONE_REQUEST_DISPATCH_NEW = """    if one_request:
        return run_chain_batch_one_request(engine, specs, encode=encode, decode=decode,
                                           budget=budget, query_stride=query_stride,
                                           block_cap=block_cap)
    limit = max_active if max_active > 0 else len(specs)
"""
ONE_REQUEST_FN = ONE_REQUEST_MARK + '''
# With the always-continue policy, draw_steps never acts at a block end: its
# per-block requests (at most block_cap tokens each) exist only to record where
# blocks end, and each one re-reads the whole context. Here each chain is one
# vLLM request instead, which samples from the same distribution (prompts are
# token ids, so nothing is re-tokenised). Its tokens are then replayed through
# draw_steps, so block ends, query points and the result are exactly what the
# block-by-block loop records for those tokens. Per-request seeds differ, so the
# chains are new draws, not the ones the block-by-block loop would sample.


class _ReplayPastEnd(RuntimeError):
    pass


class _RecordedChainEngine:
    """Serve draw_steps' requests from one chain's recorded tokens, stopping as
    vLLM does: at the first token whose text completes a stop string (text cut
    after it, token kept), on an end-of-sequence token (its text not shown), or
    at max_tokens."""

    def __init__(self, decode, prompt_len, ids, eos_ids):
        self.decode, self.prompt_len = decode, prompt_len
        self.ids, self.eos_ids = list(ids), set(eos_ids)

    def generate_batch(self, requests):
        return [self._one(r) for r in requests]

    def _one(self, req):
        from types import SimpleNamespace

        start = len(req.prompt_ids) - self.prompt_len
        ids = self.ids
        if start >= len(ids):
            raise _ReplayPastEnd(f"asked for token {start} of a {len(ids)}-token chain")
        taken = []
        for i in range(start, min(len(ids), start + req.max_tokens)):
            taken.append(ids[i])
            if i == len(ids) - 1 and ids[i] in self.eos_ids:
                return SimpleNamespace(token_ids=taken, text=self.decode(taken[:-1]),
                                       finish_reason="stop", stop_reason=ids[i], hit_eos=True)
            text = self.decode(taken)
            hits = [text.find(s) + len(s) for s in req.stop if s in text]
            if hits:
                return SimpleNamespace(token_ids=taken, text=text[:min(hits)],
                                       finish_reason="stop", stop_reason=None, hit_eos=False)
        return SimpleNamespace(token_ids=taken, text=self.decode(taken), finish_reason="length",
                               stop_reason=None, hit_eos=False)


def run_chain_batch_one_request(engine, specs, *, encode, decode, budget: int,
                                query_stride: int, block_cap: int):
    import functools
    import json as _json
    import sys as _sys
    import time as _time

    from rl_controller.rollout_contract import CLOSE_TAG, block_stop_strings, sampling_config

    # Stop once thinking has closed and the next block has ended, which is
    # where draw_steps stops too; a chain that closes thinking without a blank
    # line after it runs on to end-of-sequence, which only costs tokens.
    stops = [CLOSE_TAG + s for s in block_stop_strings()]
    requests = [GenRequest(prompt_ids=list(s["prompt_ids"]), stop=stops,
                           include_stop_str_in_output=True, max_tokens=budget,
                           seed_key=(s["seed_key"], "one-request")) for s in specs]
    t0 = _time.time()
    llm = getattr(engine, "_llm", None)
    if llm is not None:
        from tqdm import tqdm
        from vllm import TokensPrompt

        prompts = [TokensPrompt(prompt_token_ids=r.prompt_ids) for r in requests]
        params = [engine._params(stop=r.stop, include_stop_str_in_output=True,
                                 max_tokens=r.max_tokens, seed_key=r.seed_key) for r in requests]
        bar = functools.partial(tqdm, mininterval=300, file=_sys.stdout, ascii=True)
        outs = [engine._wrap(o.outputs[0]) for o in llm.generate(prompts, params, use_tqdm=bar)]
    else:
        outs = engine.generate_batch(requests)
    gen_s = _time.time() - t0

    sampling = getattr(engine, "_sampling", None) or sampling_config()
    eos_ids = sampling["eos_token_id"]
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
    t1 = _time.time()
    played, redo = [], []
    for k, (spec, out) in enumerate(zip(specs, outs)):
        recorded = _RecordedChainEngine(decode, len(spec["prompt_ids"]), out.token_ids, eos_ids)
        try:
            played.append(run_chain_batch(recorded, [spec], encode=encode, budget=budget,
                                          query_stride=query_stride, block_cap=block_cap)[0])
        except _ReplayPastEnd:
            played.append(None)
            redo.append(k)
    replay_s = _time.time() - t1
    if redo:
        # The replay needed tokens past the recorded end; sample these chains
        # block by block, as BELT does.
        again = run_chain_batch(engine, [specs[k] for k in redo], encode=encode, budget=budget,
                                query_stride=query_stride, block_cap=block_cap, max_active=32)
        for k, entry in zip(redo, again):
            played[k] = entry
    generated = sum(len(o.token_ids) for o in outs)
    print(_json.dumps({"one_request": {
        "chains": len(specs), "generated_tokens": generated, "gen_s": round(gen_s, 1),
        "tokens_per_s": round(generated / max(gen_s, 1e-9), 1),
        "replay_s": round(replay_s, 1), "block_by_block_fallback": len(redo),
        "kept_tokens": sum(len(a["result"].completion_ids) for a in played)}}), flush=True)
    return played
'''
PROBE_ONE_REQUEST_OLD = """                                 log_every=int(_os.environ.get("BELT_LOG_EVERY_CHAINS") or 0))
"""
PROBE_ONE_REQUEST_NEW = """                                 log_every=int(_os.environ.get("BELT_LOG_EVERY_CHAINS") or 0),
                                 one_request=_os.environ.get("BELT_ONE_REQUEST_CHAINS") == "1",
                                 decode=decode)
"""

LOAD_BACKEND_OLD = """    llm = LLM(model=(args.model_path or args.model),"""
LOAD_BACKEND_NEW = """    from rl_controller.rollout_contract import active_prompt_contract as _contract
    if _contract() == "qwen3.6-35b-a3b-fp8-native-thinking-v1":
        # Vision-language checkpoint, text-only prompts: no encoder budget.
        extra["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(model=(args.model_path or args.model),"""

FALLBACK_EDITS = (
    (
        "    QWEN_CONTRACT,\n    contract_identity,",
        "    QWEN_CONTRACT,\n    QWEN36_CONTRACT,\n    contract_identity,",
    ),
    (
        'CONTRACTS = {"qwen3": QWEN_CONTRACT, "olmo3": OLMO3_CONTRACT}\n'
        'DEFAULT_MAX_MODEL_LEN = {"qwen3": 40960, "olmo3": 20480}',
        'CONTRACTS = {"qwen3": QWEN_CONTRACT, "olmo3": OLMO3_CONTRACT, '
        '"qwen36": QWEN36_CONTRACT}\n'
        'DEFAULT_MAX_MODEL_LEN = {"qwen3": 40960, "olmo3": 20480, "qwen36": 20480}',
    ),
)

POOL_OLD = """    "gpqa_diamond": {"dataset": "Idavidrein/gpqa", "config": "gpqa_diamond", "split": "train",
                     "field": "Question", "gated": True},
}"""
POOL_NEW = """    "gpqa_diamond": {"dataset": "Idavidrein/gpqa", "config": "gpqa_diamond", "split": "train",
                     "field": "Question", "gated": True},
    # Added for Qwen3.6: ThinkBooster evaluates on Minerva Math.
    "minerva_math": {"dataset": "test-time-compute/test_minerva_math", "config": None,
                     "split": "test", "field": "question"},
}"""

RUNTIME_OLD = """    attention = configure_attention(attention_mode(getattr(args, "attention", None)))
    model = AutoModel.from_pretrained(model_ref, revision=None if args.model_path else revision,
                                     cache_dir=args.hf_cache, torch_dtype=dtype,
                                     **attention).to(device)"""
RUNTIME_NEW = """    attention = configure_attention(attention_mode(getattr(args, "attention", None)))
    if backbone["model"] == "Qwen/Qwen3.6-35B-A3B-FP8":
        # Text-only, truncated to the deepest feature layer, FP8 dequantized to
        # bf16. Nothing after that layer is read, so the features are unchanged.
        # QWEN36_BACKBONE_DEVICE puts the backbone on its own GPU: 24 bf16 layers
        # take ~40 GB of a 48 GB card, and the reader's causal encoder needs
        # ~6.5 GB more for one 14k-token chain. Features reach the head through
        # the CPU (MultilayerHiddenStateProvider), so the split changes nothing.
        import os as _os
        from transformers import AutoConfig, AutoModelForCausalLM, FineGrainedFP8Config
        backbone_device = torch.device(_os.environ.get("QWEN36_BACKBONE_DEVICE") or device)
        layers = tuple(saved_config["layers"]) if saved_config is not None else tuple(args.uhead_layers)
        keep = max(layers)
        hf_config = AutoConfig.from_pretrained(model_ref, revision=None if args.model_path else revision,
                                               cache_dir=args.hf_cache)
        text = getattr(hf_config, "text_config", hf_config)
        if not 1 <= keep <= text.num_hidden_layers:
            raise ValueError(f"feature layer {keep} is outside 1..{text.num_hidden_layers}")
        text.num_hidden_layers = keep
        if getattr(text, "layer_types", None):
            text.layer_types = list(text.layer_types[:keep])
        model = AutoModelForCausalLM.from_pretrained(
            model_ref, revision=None if args.model_path else revision, config=hf_config,
            cache_dir=args.hf_cache, dtype=dtype, device_map={"": backbone_device},
            quantization_config=FineGrainedFP8Config(dequantize=True), **attention)
        print(json.dumps({"qwen36_backbone": {"layers_loaded": keep, "fp8_dequantized": True,
                                              "dtype": str(dtype), "class": type(model).__name__,
                                              "backbone_device": str(backbone_device),
                                              "head_device": str(device)}}),
              flush=True)
    else:
        model = AutoModel.from_pretrained(model_ref, revision=None if args.model_path else revision,
                                         cache_dir=args.hf_cache, torch_dtype=dtype,
                                         **attention).to(device)"""


HEAD_OLD = """    def _encode(self, seq, padding, seq_len):
        mask = self._build_causal_step_mask(seq_len, seq.device)
"""
HEAD_NEW = """    def _encode(self, seq, padding, seq_len):
        mask = self._build_causal_step_mask(seq_len, seq.device)
        from rl_controller.rollout_contract import active_prompt_contract as _contract
        if _contract() == "qwen3.6-35b-a3b-fp8-native-thinking-v1":
            # Qwen3.6 chains reach 16k tokens; two paths in nn.MultiheadAttention
            # would build a per-head [B*H, T, T] mask (8 GiB at T = 16k):
            # - the eval/no-grad "fast path" (the frozen copy and scoring) always
            #   merges the masks, so it is switched off;
            # - an all-False key-padding mask is merged in too, so it is dropped.
            # What remains is the causal mask alone, run by SDPA without
            # materializing attention. Outputs move by fp32 rounding (3.6e-5 on
            # a 4k-token chain).
            torch.backends.mha.set_fastpath_enabled(False)
            if padding is not None and not bool(padding.any()):
                padding = None
"""


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(
            f"{path}: expected one match for the edit, found {text.count(old)}"
        )
    path.write_text(text.replace(old, new))


def apply_one_request(root: Path) -> None:
    """Edit 9 on a checkout that already has edits 1 to 8."""
    for package in PACKAGES:
        pkg = root / package
        chain = pkg / "single_cot.py"
        replace_once(chain, ONE_REQUEST_SIG_OLD, ONE_REQUEST_SIG_NEW)
        replace_once(chain, ONE_REQUEST_DISPATCH_OLD, ONE_REQUEST_DISPATCH_NEW)
        chain.write_text(chain.read_text().rstrip("\n") + "\n\n\n" + ONE_REQUEST_FN)
        probe = pkg / "scripts" / "single_cot_probe.py"
        if probe.exists():
            replace_once(probe, PROBE_ONE_REQUEST_OLD, PROBE_ONE_REQUEST_NEW)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("belt_root", type=Path)
    parser.add_argument("--add-one-request", action="store_true",
                        help="apply only edit 9 to a checkout patched with edits 1 to 8")
    args = parser.parse_args()
    root = args.belt_root.resolve()
    if args.add_one_request:
        chain = root / PACKAGES[0] / "single_cot.py"
        if MARK not in (root / PACKAGES[0] / "rollout_contract.py").read_text():
            raise SystemExit("this checkout does not have edits 1 to 8")
        if ONE_REQUEST_MARK in chain.read_text():
            raise SystemExit("this checkout already has edit 9")
        apply_one_request(root)
        print("added edit 9 to", root)
        return 0

    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != BELT_COMMIT:
        raise SystemExit(f"BELT checkout is at {head}, expected {BELT_COMMIT}")
    if MARK in (root / PACKAGES[0] / "rollout_contract.py").read_text():
        raise SystemExit("this checkout is already patched")

    for package in PACKAGES:
        pkg = root / package
        shutil.copyfile(HERE / "qwen36_contract.py", pkg / "qwen36_contract.py")
        contract = pkg / "rollout_contract.py"
        contract.write_text(
            contract.read_text().rstrip("\n") + "\n\n\n" + CONTRACT_BLOCK
        )
        probe = pkg / "scripts" / "single_cot_probe.py"
        if probe.exists():
            replace_once(probe, LOAD_BACKEND_OLD, LOAD_BACKEND_NEW)
            replace_once(probe, PROBE_CALL_OLD, PROBE_CALL_NEW)
        chain = pkg / "single_cot.py"
        replace_once(chain, CHAIN_BATCH_OLD, CHAIN_BATCH_NEW)
        for old, new in RUNNER_EDITS:
            replace_once(pkg / "rollout_runner.py", old, new)
        for old, new in BATCH_EDITS:
            replace_once(pkg / "rollout_batch.py", old, new)
    apply_one_request(root)

    fallback = root / "rl_controller/scripts/belt_natural_fallback.py"
    for old, new in FALLBACK_EDITS:
        replace_once(fallback, old, new)
    replace_once(
        root / "rl_controller/scripts/iclr_build_math_pool.py", POOL_OLD, POOL_NEW
    )
    replace_once(
        root / "scripts/source_variants/olmo_belt/rl_controller/multilayer_runtime.py",
        RUNTIME_OLD,
        RUNTIME_NEW,
    )
    replace_once(
        root / "scripts/source_variants/olmo_belt/rl_controller/multilayer_uhead.py",
        HEAD_OLD,
        HEAD_NEW,
    )

    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--stat"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    print(diff)
    print("patched", root, "at", BELT_COMMIT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
