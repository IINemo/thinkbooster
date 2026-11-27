"""Add the Qwen3.6-35B-A3B contract to a BELT checkout, without changing BELT's paths.

BELT (github.com/EdisonNi-hku/BELT, pinned at ``BELT_COMMIT``) ships its Qwen3
and OLMo-3 recipes as code snapshots. Qwen3.6 has no ReProbe head, so it
follows the OLMo path: ``scripts/source_variants/olmo_code`` generates and
labels chains, ``scripts/source_variants/olmo_belt`` trains the self-contained
reader, and the root package builds the question pool and runs the
natural-answer fallback. This script makes four kinds of edit, each asserted
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
   features.

    python scripts/belt_qwen36/apply_belt_patch.py /path/to/BELT

It refuses a checkout that is not at ``BELT_COMMIT`` or is already patched.
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
        from transformers import AutoConfig, AutoModelForCausalLM, FineGrainedFP8Config
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
            cache_dir=args.hf_cache, dtype=dtype, device_map={"": device},
            quantization_config=FineGrainedFP8Config(dequantize=True), **attention)
        print(json.dumps({"qwen36_backbone": {"layers_loaded": keep, "fp8_dequantized": True,
                                              "dtype": str(dtype), "class": type(model).__name__}}),
              flush=True)
    else:
        model = AutoModel.from_pretrained(model_ref, revision=None if args.model_path else revision,
                                         cache_dir=args.hf_cache, torch_dtype=dtype,
                                         **attention).to(device)"""


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(
            f"{path}: expected one match for the edit, found {text.count(old)}"
        )
    path.write_text(text.replace(old, new))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("belt_root", type=Path)
    args = parser.parse_args()
    root = args.belt_root.resolve()

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
