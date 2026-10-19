"""Opt-in, fixed-input GPU software validation; never an accuracy benchmark.

Run ``hf``, ``vllm``, then ``hf-generated`` in three separate Python processes.
The policies below are fixed, unfitted software fixtures. Their coefficients,
cases and tolerances must not be adjusted after observing validation results.
"""

import argparse
import bisect
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from unittest.mock import patch

import torch

from thinkbooster.belt import BeltStoppingPolicy
from thinkbooster.belt import runtime as belt_runtime
from thinkbooster.belt.runtime import (
    BACKBONE,
    BACKBONE_REVISION,
    CHECKPOINT_REVISIONS,
    DEFAULT_CHECKPOINT,
    BeltRuntime,
    checkpoint_manifest,
)

SCHEMA = "thinkbooster-belt-vllm-validation/v1"
PUBLIC_SOURCES = (
    "thinkbooster/belt/__init__.py",
    "thinkbooster/belt/runtime.py",
    "thinkbooster/belt/policy.py",
    "thinkbooster/belt/vllm_worker.py",
    "thinkbooster/belt/vllm_wire.py",
    "thinkbooster/belt/vllm_backend.py",
    "thinkbooster/scorers/step_scorer_belt.py",
    "thinkbooster/generators/base.py",
    "thinkbooster/generators/vllm.py",
    "thinkbooster/strategies/strategy_base.py",
    "thinkbooster/strategies/strategy_belt.py",
    "thinkbooster/step_boundary_detectors/thinking/marker.py",
)
TOLERANCES = {
    "feature_relative_l2": 0.02,
    "hidden_relative_l2": 0.02,
    "belief_max_abs": 0.01,
    "policy_logit_abs": 0.05,
}
# Noise-floor control. A case that misses a fixed tolerance against the HF bf16
# reference still passes that check only when vLLM is within NOISE_FACTOR times
# the HF bf16 reference's own error against an HF fp32 forward of the same IDs.
# Both numbers are always reported; this factor is fixed like the tolerances.
NOISE_FACTOR = 2.0
FP32_REFERENCE = {
    "dtype": "float32",
    "attn_implementation": "sdpa",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "float32_matmul_precision": "highest",
}
PASS_CRITERION = (
    "per case and check: vllm_vs_hf_bf16 <= tolerance, or else "
    "vllm_vs_hf_fp32 <= noise_factor * hf_bf16_vs_hf_fp32"
)
# Each check maps to its tolerance key and the metric read from one case's
# (feature error, reader signal error) pair.
CHECK_METRICS = {
    "features": ("feature_relative_l2", lambda feature, _: feature["relative_l2"]),
    "hidden": (
        "hidden_relative_l2",
        lambda _, signal: signal["hidden"]["relative_l2"],
    ),
    "beliefs": ("belief_max_abs", lambda _, signal: signal["max_belief_abs"]),
    "fixture_policy_logits": (
        "policy_logit_abs",
        lambda _, signal: signal["policy_logit_abs"],
    ),
}
PROFILE_BUCKETS = 10
PROFILE_TOP_K = 5
HF_ARTIFACTS = ("cases.json", "hf_features.pt", "hf_fp32_features.pt")
VLLM_ARTIFACTS = (
    "generated_cases.json",
    "vllm_generated_features.pt",
    "vllm_features.pt",
)
SMOKE_CHAT = [
    {
        "role": "user",
        "content": (
            "Compute 23 times 47 by decomposing 47 into 40 plus 7. Before the final "
            "answer, discuss three independent arithmetic checks step by step."
        ),
    }
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_hashes(args, names):
    return {name: file_sha256(args.output / name) for name in names}


def verify_artifacts(args, hashes):
    for name, expected in hashes.items():
        require(Path(name).name == name, "Invalid artifact name in report")
        path = args.output / name
        require(
            path.is_file() and file_sha256(path) == expected,
            f"Artifact checksum mismatch: {name}",
        )


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def source_revision(directory):
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def source_dirty(directory):
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def software_identity():
    from rl_controller.rollout_contract import resolve_native_root
    from rl_controller.scripts import rl_train

    root = Path(__file__).resolve().parents[2]
    require(
        Path(belt_runtime.__file__).resolve() == root / "thinkbooster/belt/runtime.py",
        "Imported ThinkBooster runtime must come from the helper's checkout",
    )
    sources = {name: file_sha256(root / name) for name in PUBLIC_SOURCES}
    sources["tests/belt/validate_vllm_gpu.py"] = file_sha256(__file__)

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "vllm": package_version("vllm"),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "native_source_revision": source_revision(resolve_native_root()),
        "native_runtime_sha256": file_sha256(rl_train.__file__),
        "thinkbooster_source_revision": source_revision(root),
        "thinkbooster_source_dirty": source_dirty(root),
        "public_source_sha256": sources,
    }


def fixture_policy(checksum, *, force_stop=False):
    """Fixed coefficients exercise software interfaces; nothing is fitted."""
    return BeltStoppingPolicy(
        checkpoint_sha256=checksum,
        feature_space="B",
        weights=(0, 0, 0, 0) if force_stop else (0.125, -0.25, 0.375, -0.5),
        intercept=1.0 if force_stop else 0.1,
        means=(0, 0, 0, 0),
        stds=(1, 1, 1, 1),
        threshold=0,
    )


def policy_description(policy, purpose):
    parameters = asdict(policy)
    for name in ("weights", "means", "stds"):
        parameters[name] = list(parameters[name])
    return {"fitted": False, "purpose": purpose, "parameters": parameters}


def make_cases(tokenizer):
    chat = [
        {
            "role": "user",
            "content": "Solve the problem carefully: if x=5, find x squared.",
        }
    ]
    prompt = list(
        tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    )
    chains = [
        ["Okay, so x = 5.\n"],
        ["Okay, so x = 5.\n", "\nSo the area is 25.\n"],
        ["Okay, so x = 5.\n", "\nSo the area is 25.\n", "\nLet me verify this."],
        ["We check α + β = 12.\n", "\nTherefore, the answer is 十二.\n"],
        [f"For integer i={i}, summing i and i+1 gives {2*i+1}.\n\n" for i in range(64)],
    ]
    cases = []
    for index, steps in enumerate(chains):
        encoded = tokenizer(
            "".join(steps), add_special_tokens=False, return_offsets_mapping=True
        )
        starts = [start for start, _ in encoded["offset_mapping"]]
        ends, cut = [], 0
        for step in steps:
            cut += len(step)
            end = bisect.bisect_left(starts, cut)
            if end > (ends[-1] if ends else 0):
                ends.append(end)
        cases.append(
            {
                "name": f"fixed_{index}",
                "prompt": prompt,
                "completion": encoded["input_ids"],
                "ends": ends,
            }
        )
    return cases


class StoredFeatures:
    """Replay measured HF features through the same native head-only load path."""

    chunk_size = 0

    def __init__(self, cases, features):
        self.rows = {
            tuple(case["prompt"] + case["completion"]): value
            for case, value in zip(cases, features)
        }

    def get_features(self, sequences, device):
        require(len(sequences) == 1, "Stored feature replay requires one sequence")
        value = self.rows[tuple(sequences[0])].to(device).unsqueeze(0)
        attention = torch.ones(value.shape[:2], dtype=torch.long, device=device)
        return value, attention, [0], []


@contextmanager
def forbid_hf_backbone_loads():
    """Make accidental second-backbone construction a failing software check."""
    from rl_controller.scripts import rl_train
    from transformers import AutoModel, AutoModelForCausalLM

    def forbidden(*args, **kwargs):
        raise RuntimeError(
            "Injected BELT scoring attempted an HF backbone/runtime load"
        )

    with ExitStack() as stack:
        for owner, name in (
            (AutoModel, "from_pretrained"),
            (AutoModelForCausalLM, "from_pretrained"),
            (rl_train, "_load_head_runtime"),
        ):
            stack.enter_context(patch.object(owner, name, side_effect=forbidden))
        yield


def score_cases(provider, tokenizer, config, cases, args):
    with forbid_hf_backbone_loads():
        runtime = BeltRuntime.from_pretrained(
            feature_provider=provider,
            tokenizer=tokenizer,
            base_config=config,
            device="cuda:0",
            dtype="bfloat16",
            cache_dir=str(args.cache),
        )
        try:
            require(
                runtime.checkpoint_sha256 == args.checksum,
                "Probe checksum differs from pinned manifest",
            )
            policy = fixture_policy(runtime.checkpoint_sha256)
            values = []
            for case in cases:
                signal = runtime.score_token_ids(
                    case["prompt"], case["completion"], case["ends"]
                )
                values.append(
                    {
                        "hidden": list(signal.hidden),
                        "beliefs": dict(signal.beliefs),
                        "policy_logit": policy.decision_logit(signal),
                        "checkpoint_sha256": signal.checkpoint_sha256,
                    }
                )
            return values
        finally:
            runtime.cleanup()


def report_identity(args):
    return {
        "schema": SCHEMA,
        "software_test_only": True,
        "backbone": BACKBONE,
        "revision": BACKBONE_REVISION,
        "checkpoint_repo": DEFAULT_CHECKPOINT,
        "checkpoint_revision": CHECKPOINT_REVISIONS[DEFAULT_CHECKPOINT],
        "checkpoint_seed": 0,
        "checkpoint_sha256": args.checksum,
        "software": software_identity(),
        "tolerances": dict(TOLERANCES),
        "noise_floor_control": {
            "noise_factor": NOISE_FACTOR,
            "fp32_reference": dict(FP32_REFERENCE),
            "pass_criterion": PASS_CRITERION,
        },
        "policy_fixture": policy_description(
            fixture_policy(args.checksum),
            "reader/logit parity; not a fitted stopping policy",
        ),
    }


def verify_report_identity(report, args):
    expected = report_identity(args)
    for key in (
        "schema",
        "backbone",
        "revision",
        "checkpoint_repo",
        "checkpoint_revision",
        "checkpoint_seed",
        "checkpoint_sha256",
        "tolerances",
        "noise_floor_control",
        "policy_fixture",
    ):
        require(
            report[key] == expected[key], f"Reference fixture identity mismatch: {key}"
        )
    for key in (
        "python",
        "torch",
        "transformers",
        "vllm",
        "gpu",
        "cuda_runtime",
        "native_runtime_sha256",
        "public_source_sha256",
    ):
        require(
            report["software"][key] == expected["software"][key],
            f"Reference software/source/GPU mismatch: {key}",
        )


@contextmanager
def strict_fp32_matmul():
    """The fp32 noise-floor reference must not silently run matmuls in TF32."""
    saved = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    torch.set_float32_matmul_precision(FP32_REFERENCE["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = FP32_REFERENCE["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = FP32_REFERENCE["cudnn_allow_tf32"]
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(saved[0])
        torch.backends.cuda.matmul.allow_tf32 = saved[1]
        torch.backends.cudnn.allow_tf32 = saved[2]


def hf_backbone_features(args, cases, dtype=torch.bfloat16):
    """Forward exact saved IDs and free the HF backbone before reader loading."""
    from transformers import AutoModelForCausalLM

    require(
        dtype in (torch.bfloat16, torch.float32), "HF reference is bf16 or fp32 only"
    )
    with ExitStack() as stack:
        if dtype == torch.float32:
            stack.enter_context(strict_fp32_matmul())
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=dtype,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .to("cuda:0")
            .eval()
        )
        require(
            dtype != torch.float32
            or all(parameter.dtype == dtype for parameter in model.parameters()),
            "HF fp32 noise-floor backbone did not load every parameter in float32",
        )
        features, timings = [], []
        for case in cases:
            ids = torch.tensor([case["prompt"] + case["completion"]], device="cuda:0")
            torch.cuda.synchronize()
            before = time.perf_counter()
            with torch.inference_mode():
                output = model(ids, output_hidden_states=True, use_cache=False)
                hidden = output.hidden_states[16]
                require(hidden.dtype == dtype, "HF residual dtype differs from model")
                features.append(hidden[0, :-1].float().cpu())
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - before)
            del output, hidden, ids
        del model
    gc.collect()
    torch.cuda.empty_cache()
    return features, timings


def noise_floor(cases, bf16_features, fp32_features, bf16_signals, fp32_signals):
    """Per-case error of the HF bf16 reference against the HF fp32 reference."""
    require(
        len(cases)
        == len(bf16_features)
        == len(fp32_features)
        == len(bf16_signals)
        == len(fp32_signals),
        "Noise floor needs one bf16 and one fp32 reference per fixed case",
    )
    signal_errors = signal_error_metrics(fp32_signals, bf16_signals)
    return [
        {
            "name": case["name"],
            "features": error_metrics(fp32, bf16),
            "signals": signal,
        }
        for case, bf16, fp32, signal in zip(
            cases, bf16_features, fp32_features, signal_errors
        )
    ]


def hf_phase(args):
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    cases = make_cases(tokenizer)
    write_json(args.output / "cases.json", cases)
    start = time.perf_counter()
    features, timings = hf_backbone_features(args, cases)
    torch.save(features, args.output / "hf_features.pt")
    # Noise-floor control: same weights, IDs and attention, float32 without TF32.
    # The bf16 backbone is already freed; this one is freed before reader loading.
    fp32_features, fp32_timings = hf_backbone_features(args, cases, torch.float32)
    torch.save(fp32_features, args.output / "hf_fp32_features.pt")
    signals = score_cases(
        StoredFeatures(cases, features), tokenizer, config, cases, args
    )
    fp32_signals = score_cases(
        StoredFeatures(cases, fp32_features), tokenizer, config, cases, args
    )
    write_json(
        args.output / "hf.json",
        {
            **report_identity(args),
            "phase": "hf",
            "status": "passed",
            "overall_status": "pending_vllm_and_generated_parity",
            "input_artifacts": artifact_hashes(args, HF_ARTIFACTS),
            "signals": signals,
            "fp32_signals": fp32_signals,
            "noise_floor": noise_floor(
                cases, features, fp32_features, signals, fp32_signals
            ),
            "lengths": [
                len(case["prompt"]) + len(case["completion"]) for case in cases
            ],
            "head_only_load_guard_passed": True,
            "feature_seconds": timings,
            "fp32_feature_seconds": fp32_timings,
            "phase_seconds": time.perf_counter() - start,
        },
    )
    print("HF_REFERENCE_OK", flush=True)


def error_metrics(reference, actual):
    require(
        reference.shape == actual.shape,
        f"Feature shape differs: {reference.shape} versus {actual.shape}",
    )
    require(
        bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()),
        "Nonfinite features",
    )
    reference, actual = reference.double().reshape(-1), actual.double().reshape(-1)
    diff = actual - reference
    return {
        "max_abs": diff.abs().max().item(),
        "rms": diff.square().mean().sqrt().item(),
        "relative_l2": (diff.norm() / reference.norm().clamp_min(1e-12)).item(),
        "cosine": torch.nn.functional.cosine_similarity(
            reference, actual, dim=0
        ).item(),
    }


def signal_error_metrics(reference, actual):
    require(len(reference) == len(actual), "Reader signal count mismatch")
    result = []
    for left, right in zip(reference, actual):
        require(
            left["checkpoint_sha256"] == right["checkpoint_sha256"],
            "Probe checksum mismatch",
        )
        result.append(
            {
                "hidden": error_metrics(
                    torch.tensor(left["hidden"]), torch.tensor(right["hidden"])
                ),
                "max_belief_abs": max(
                    abs(left["beliefs"][key] - right["beliefs"][key])
                    for key in left["beliefs"]
                ),
                "policy_logit_abs": abs(left["policy_logit"] - right["policy_logit"]),
            }
        )
    return result


def parity_checks(feature_errors, signal_errors):
    """Fixed tolerances only, every case."""
    require(
        bool(feature_errors) and len(feature_errors) == len(signal_errors),
        "Parity requires a nonempty matched feature/signal bank",
    )
    return {
        check: all(
            metric(feature, signal) <= TOLERANCES[key]
            for feature, signal in zip(feature_errors, signal_errors)
        )
        for check, (key, metric) in CHECK_METRICS.items()
    }


def noise_aware_checks(
    cases, feature_errors, signal_errors, fp32_feature_errors, fp32_signal_errors, floor
):
    """Fixed tolerance first; otherwise compare against the bf16 noise floor.

    ``feature_errors``/``signal_errors`` are vLLM versus HF bf16,
    ``fp32_*`` are vLLM versus HF fp32 and ``floor`` is ``noise_floor(...)``.
    Every case records both numbers and which criterion, if any, passed.
    """
    require(
        bool(cases)
        and len(cases)
        == len(feature_errors)
        == len(signal_errors)
        == len(fp32_feature_errors)
        == len(fp32_signal_errors)
        == len(floor),
        "Noise-aware parity requires matched nonempty per-case errors",
    )
    checks = {check: True for check in CHECK_METRICS}
    case_checks, via_floor = [], []
    for index, case in enumerate(cases):
        require(floor[index]["name"] == case["name"], "Noise floor case order differs")
        row = {"name": case["name"]}
        for check, (key, metric) in CHECK_METRICS.items():
            versus_bf16 = metric(feature_errors[index], signal_errors[index])
            versus_fp32 = metric(fp32_feature_errors[index], fp32_signal_errors[index])
            floor_value = metric(floor[index]["features"], floor[index]["signals"])
            bound = NOISE_FACTOR * floor_value
            if versus_bf16 <= TOLERANCES[key]:
                criterion = "tolerance"
            elif versus_fp32 <= bound:
                criterion = "noise_floor"
                via_floor.append(f"{case['name']}:{check}")
            else:
                criterion = "failed"
                checks[check] = False
            row[check] = {
                "passed_by": criterion,
                "vllm_vs_hf_bf16": versus_bf16,
                "tolerance": TOLERANCES[key],
                "vllm_vs_hf_fp32": versus_fp32,
                "hf_bf16_vs_hf_fp32": floor_value,
                "noise_bound": bound,
                "vllm_fp32_to_floor_ratio": (
                    versus_fp32 / floor_value if floor_value > 0 else None
                ),
            }
        case_checks.append(row)
    return checks, case_checks, via_floor


def position_profile(reference, actual, token_ids, prompt_length):
    """Row-wise relative L2 by position bucket plus the worst positions.

    Row ``p`` is the residual after reading ``token_ids[p]`` (``[:-1]`` alignment).
    """
    require(
        reference.ndim == 2
        and reference.shape == actual.shape
        and reference.shape[0] > 0
        and len(token_ids) == reference.shape[0] + 1,
        "Position profile needs matched [T-1, H] features for T token IDs",
    )
    reference, actual = reference.double(), actual.double()
    diff_norms = (actual - reference).norm(dim=1)
    reference_norms = reference.norm(dim=1)
    rows = diff_norms / reference_norms.clamp_min(1e-12)
    count = rows.shape[0]
    parts = min(PROFILE_BUCKETS, count)
    buckets = []
    for part in range(parts):
        start, stop = part * count // parts, (part + 1) * count // parts
        buckets.append(
            {
                "start": start,
                "stop": stop,
                "relative_l2": (
                    diff_norms[start:stop].norm()
                    / reference_norms[start:stop].norm().clamp_min(1e-12)
                ).item(),
                "max_row_relative_l2": rows[start:stop].max().item(),
            }
        )
    values, positions = rows.topk(min(PROFILE_TOP_K, count))
    return {
        "rows": count,
        "prompt_rows": prompt_length,
        "row_relative_l2": {
            "median": rows.median().item(),
            "p90": torch.quantile(rows, 0.9).item(),
            "max": rows.max().item(),
        },
        "buckets": buckets,
        "worst_positions": [
            {
                "position": position,
                "token_id": token_ids[position],
                "completion_index": (
                    position - prompt_length if position >= prompt_length else None
                ),
                "relative_l2": value,
            }
            for value, position in zip(values.tolist(), positions.tolist())
        ],
    }


def case_position_profiles(cases, bf16_features, fp32_features, vllm_features):
    profiles = []
    for case, bf16, fp32, vllm in zip(
        cases, bf16_features, fp32_features, vllm_features
    ):
        ids, prompt_length = case["prompt"] + case["completion"], len(case["prompt"])
        profiles.append(
            {
                "name": case["name"],
                "vllm_vs_hf_bf16": position_profile(bf16, vllm, ids, prompt_length),
                "vllm_vs_hf_fp32": position_profile(fp32, vllm, ids, prompt_length),
                "hf_bf16_vs_hf_fp32": position_profile(fp32, bf16, ids, prompt_length),
            }
        )
    return profiles


def strategy_smoke(backend, config, args):
    from thinkbooster.belt.blocks import ends_on_block
    from thinkbooster.generators.vllm import VLLMStepGenerator
    from thinkbooster.scorers.step_scorer_belt import StepScorerBELT
    from thinkbooster.step_boundary_detectors.thinking import ThinkingMarkerDetector
    from thinkbooster.strategies.strategy_belt import StrategyBELT

    # BELT steps are blank-line blocks: no minimum, room for a whole block.
    detector = ThinkingMarkerDetector(min_step_tokens=0, max_step_tokens=128)
    generator = VLLMStepGenerator(
        model=backend,
        thinking_mode=True,
        detector=detector,
        max_new_tokens=64,
        temperature=0,
        top_p=1,
        top_k=-1,
        max_context_budget=8192,
        disable_thinking_mode=False,
        answer_patterns=[],
        seed=0,
    )
    scorer = StepScorerBELT(
        feature_provider=backend,
        tokenizer=backend.get_tokenizer(),
        base_config=config,
        device="cuda:0",
        dtype="bfloat16",
        cache_dir=str(args.cache),
    )
    policy = fixture_policy(args.checksum, force_stop=True)
    with forbid_hf_backbone_loads():
        # Plumbing smoke: stop at the earliest block end, block 1 included.
        strategy = StrategyBELT(generator, scorer, policy, max_steps=8, min_steps=1)
        result = strategy.generate_trajectory(SMOKE_CHAT)
        texts = [step.raw_text or step.text for step in result["steps"]]
        require(
            scorer.runtime.feature_provider is backend,
            "Strategy did not use its generation feature provider",
        )
        stats = backend.stats
        smoke = {
            "chat": SMOKE_CHAT,
            "policy_fixture": policy_description(
                policy, "force stop at first eligible query"
            ),
            "generator": {
                "min_step_tokens": 0,
                "max_step_tokens": 128,
                "answer_max_tokens": 64,
                "seed": 0,
            },
            "head_only_load_guard_passed": True,
            "scorer_metadata": scorer.metadata,
            "backend_metadata": backend.metadata,
            "token_stats": result["token_stats"],
            "completed": result["completed"],
            "completion_reason": result["completion_reason"],
            "reasoning_steps": result["reasoning_steps"],
            "metadata": result["metadata"],
            "step_texts": texts,
            "checks": {
                "queried_real_reader": stats["queries"] >= 1
                and bool(result["metadata"]["decisions"]),
                "fixture_triggered_stop": any(
                    row["stop"] for row in result["metadata"]["decisions"]
                ),
                "captured_generation": stats["generation_calls"] >= 2
                and stats["generation_captured_tokens"] > 0,
                "cost_scope_preserved": result["metadata"][
                    "scoring_cost_in_token_stats"
                ]
                is False,
                "scoring_backend": result["metadata"]["scoring_backend"] == "vllm",
                "decisions_at_block_ends": all(
                    ends_on_block("".join(texts[: row["step"]]))
                    for row in result["metadata"]["decisions"]
                ),
            },
        }
        strategy.cleanup()
    return smoke


@contextmanager
def owned_vllm_engine(kwargs):
    """This standalone helper owns its engine; shared providers never close it."""
    from vllm import LLM

    llm = LLM(**kwargs)
    try:
        yield llm
    finally:
        original = sys.exc_info()[1]
        try:
            # vLLM 0.12 EngineCoreClient API (both MP and in-process clients).
            llm.llm_engine.engine_core.shutdown()
        except BaseException as error:
            if original is None:
                raise
            message = f"Owned vLLM engine shutdown also failed: {error!r}"
            if hasattr(original, "add_note"):
                original.add_note(message)
            else:
                print(message, file=sys.stderr, flush=True)


def vllm_phase(args):
    require(package_version("vllm") == "0.12.0", "This fixture requires vllm==0.12.0")
    from transformers import AutoConfig
    from vllm import SamplingParams

    from thinkbooster.belt.vllm_backend import BeltVLLMBackend

    require(
        all((args.output / name).is_file() for name in HF_ARTIFACTS + ("hf.json",)),
        "Run the HF phase first in a separate process",
    )
    hf = json.loads((args.output / "hf.json").read_text())
    verify_report_identity(hf, args)
    require(hf["status"] == "passed", "HF reference phase did not pass")
    require(
        set(hf["input_artifacts"]) == set(HF_ARTIFACTS),
        "HF reference artifacts differ from the bf16 + fp32 reference set",
    )
    verify_artifacts(args, hf["input_artifacts"])
    cases = json.loads((args.output / "cases.json").read_text())
    reference = torch.load(
        args.output / "hf_features.pt", map_location="cpu", weights_only=True
    )
    fp32_reference = torch.load(
        args.output / "hf_fp32_features.pt", map_location="cpu", weights_only=True
    )
    require(
        len(hf["signals"]) == len(hf["fp32_signals"]) == len(cases),
        "HF reference signal count differs from fixed cases",
    )
    require(
        len(fp32_reference) == len(cases),
        "HF fp32 reference feature count differs from the fixed cases",
    )
    start = time.perf_counter()
    llm_kwargs = {
        "model": str(args.model),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 8192,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": 0.65,
        "enforce_eager": True,
        "enable_prefix_caching": not args.disable_prefix_caching,
        "enable_chunked_prefill": True,
        "seed": 0,
        "worker_extension_cls": "thinkbooster.belt.vllm_worker.BeltHiddenStatesExtension",
    }
    with owned_vllm_engine(llm_kwargs) as llm:
        backend = BeltVLLMBackend(llm)
        tokenizer = backend.get_tokenizer()
        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        require(
            make_cases(tokenizer) == cases,
            "Pinned tokenizer does not reproduce the fixed cases",
        )
        require(
            len(reference) == len(cases),
            "HF reference feature count differs from the fixed cases",
        )
        features = []
        for case in cases:
            ids = case["prompt"] + case["completion"]
            output, attention, kept, skipped = backend.get_features([ids], "cpu")
            require(
                kept == [0] and not skipped and bool(attention.all()),
                "Invalid feature coverage",
            )
            features.append(output[0].contiguous())
        # Saved before any comparison so a failing run can be diagnosed offline.
        torch.save(features, args.output / "vllm_features.pt")
        cold_stats = backend.stats
        for case, expected in zip(cases, features):
            value, *_ = backend.get_features(
                [case["prompt"] + case["completion"]], "cpu"
            )
            torch.testing.assert_close(value[0], expected, rtol=0, atol=0)
        require(
            backend.stats["repair_calls"] == cold_stats["repair_calls"],
            "Warm cache unexpectedly repaired",
        )
        signals = score_cases(backend, tokenizer, config, cases, args)
        parity_query_metadata = backend.metadata
        feature_errors = [error_metrics(r, v) for r, v in zip(reference, features)]
        signal_errors = signal_error_metrics(hf["signals"], signals)
        fp32_feature_errors = [
            error_metrics(r, v) for r, v in zip(fp32_reference, features)
        ]
        fp32_signal_errors = signal_error_metrics(hf["fp32_signals"], signals)
        # Recomputed from the checksum-verified reference artifacts.
        floor = noise_floor(
            cases, reference, fp32_reference, hf["signals"], hf["fp32_signals"]
        )
        noise_checks, case_checks, via_floor = noise_aware_checks(
            cases,
            feature_errors,
            signal_errors,
            fp32_feature_errors,
            fp32_signal_errors,
            floor,
        )
        profiles = case_position_profiles(cases, reference, fp32_reference, features)
        backend.reset_hs_step_cache()
        params = SamplingParams(
            n=1, max_tokens=64, temperature=0, seed=0, ignore_eos=True
        )
        prompts = [
            {"prompt_token_ids": case["prompt"] + case["completion"]}
            for case in cases[:4]
        ]
        generated = backend.generate(prompts, params, use_tqdm=False)
        after_generate = backend.stats
        require(len(generated) == len(prompts), "Wrong generation batch size")
        generated_cases, generated_features = [], []
        for fixed, item in zip(cases[:4], generated):
            generation_prompt = list(item.prompt_token_ids)
            new_tokens = list(item.outputs[0].token_ids)
            require(
                generation_prompt == fixed["prompt"] + fixed["completion"],
                "Generated request prompt IDs differ from the fixed input",
            )
            require(
                len(new_tokens) == 64, "Expected exactly 64 actual generated token IDs"
            )
            completion = fixed["completion"] + new_tokens
            generated_cases.append(
                {
                    "name": "generated_" + fixed["name"],
                    "request_id": item.request_id,
                    "prompt": fixed["prompt"],
                    "completion": completion,
                    "ends": fixed["ends"] + [len(completion)],
                    "generation_prompt_token_ids": generation_prompt,
                    "generation_prompt_length": len(generation_prompt),
                    "generated_token_ids": new_tokens,
                }
            )
            ids = generation_prompt + new_tokens
            value, attention, kept, skipped = backend.get_features([ids], "cpu")
            require(
                kept == [0] and not skipped and bool(attention.all()),
                "Invalid generated feature coverage",
            )
            require(
                value.shape == (1, len(ids) - 1, 2048),
                "Generated feature alignment mismatch",
            )
            generated_features.append(value[0].contiguous())
        generated_signals = score_cases(
            backend, tokenizer, config, generated_cases, args
        )
        generation_repairs = (
            backend.stats["repair_calls"] - after_generate["repair_calls"]
        )
        generation_metadata = backend.metadata
        write_json(args.output / "generated_cases.json", generated_cases)
        torch.save(generated_features, args.output / "vllm_generated_features.pt")
        smoke = strategy_smoke(backend, config, args)
        checks = {
            **noise_checks,
            "generation_capture_reused_without_repair": generation_repairs == 0,
            "strategy_smoke": all(smoke["checks"].values()),
        }
        report = {
            **report_identity(args),
            "phase": "vllm",
            "status": "passed" if all(checks.values()) else "failed",
            "overall_status": (
                "pending_hf_generated" if all(checks.values()) else "failed"
            ),
            "input_artifacts": artifact_hashes(args, ("hf.json",) + HF_ARTIFACTS),
            "generated_artifacts": artifact_hashes(args, VLLM_ARTIFACTS),
            "generated_signals": generated_signals,
            "generation_sampling_params": {
                "n": 1,
                "max_tokens": 64,
                "temperature": 0,
                "seed": 0,
                "ignore_eos": True,
            },
            "engine_kwargs": llm_kwargs,
            "feature_errors": feature_errors,
            "signal_errors": signal_errors,
            "feature_errors_vs_hf_fp32": fp32_feature_errors,
            "signal_errors_vs_hf_fp32": fp32_signal_errors,
            "noise_floor": floor,
            "tolerance_only_checks": parity_checks(feature_errors, signal_errors),
            "case_checks": case_checks,
            "passed_via_noise_floor": via_floor,
            "position_profiles": profiles,
            "signals": signals,
            "checks": checks,
            "generation_queries": len(generated),
            "generation_query_repairs": generation_repairs,
            "cold_query_stats": cold_stats,
            "parity_query_metadata": parity_query_metadata,
            "generation_metadata": generation_metadata,
            "strategy_smoke": smoke,
            "head_only_load_guard_passed": True,
            "phase_seconds": time.perf_counter() - start,
        }
    report["owned_engine_shutdown"] = True
    write_json(args.output / "vllm.json", report)
    if via_floor:
        print(f"PASSED_VIA_NOISE_FLOOR {via_floor}; see vllm.json", flush=True)
    require(
        all(checks.values()),
        f"GPU software checks failed: {[key for key, passed in checks.items() if not passed]}; see vllm.json",
    )
    print("VLLM_CHECKS_OK_GENERATED_PARITY_PENDING", flush=True)


def hf_generated_phase(args):
    """Check numerical parity of the actual captured prefill AND decode rows."""
    from transformers import AutoConfig, AutoTokenizer

    inputs = ("hf.json",) + HF_ARTIFACTS + ("vllm.json",) + VLLM_ARTIFACTS
    require(
        all((args.output / name).is_file() for name in inputs),
        "Run hf then vllm in separate processes before hf-generated",
    )
    vllm = json.loads((args.output / "vllm.json").read_text())
    verify_report_identity(vllm, args)
    require(
        vllm["status"] == "passed" and all(vllm["checks"].values()),
        "The preceding vLLM checks must pass before generated parity",
    )
    require(
        set(vllm["input_artifacts"]) == {"hf.json", *HF_ARTIFACTS}
        and set(vllm["generated_artifacts"]) == set(VLLM_ARTIFACTS),
        "vLLM report artifact sets differ from this helper's chain",
    )
    verify_artifacts(args, vllm["input_artifacts"])
    verify_artifacts(args, vllm["generated_artifacts"])
    hf = json.loads((args.output / "hf.json").read_text())
    verify_report_identity(hf, args)
    require(
        set(hf["input_artifacts"]) == set(HF_ARTIFACTS),
        "HF reference artifacts differ from the bf16 + fp32 reference set",
    )
    verify_artifacts(args, hf["input_artifacts"])
    fixed_cases = json.loads((args.output / "cases.json").read_text())
    cases = json.loads((args.output / "generated_cases.json").read_text())
    captured = torch.load(
        args.output / "vllm_generated_features.pt",
        map_location="cpu",
        weights_only=True,
    )
    require(
        len(cases) == len(captured) == len(vllm["generated_signals"]) == 4,
        "Generated parity requires the four saved generation cases",
    )
    for fixed, case, rows in zip(fixed_cases[:4], cases, captured):
        new_tokens = case["generated_token_ids"]
        generation_prompt = fixed["prompt"] + fixed["completion"]
        require(
            len(new_tokens) == 64
            and case["prompt"] == fixed["prompt"]
            and case["generation_prompt_token_ids"] == generation_prompt
            and case["generation_prompt_length"] == len(generation_prompt)
            and case["completion"] == fixed["completion"] + new_tokens
            and case["ends"] == fixed["ends"] + [len(case["completion"])],
            "Generated token IDs or reader block-end history changed",
        )
        require(
            rows.shape == (len(generation_prompt) + 63, 2048)
            and rows.dtype == torch.float32,
            "Invalid saved next-token-aligned generated features",
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    require(make_cases(tokenizer) == fixed_cases, "Fixed tokenizer cases changed")
    start = time.perf_counter()
    features, timings = hf_backbone_features(args, cases)
    torch.save(features, args.output / "hf_generated_features.pt")
    # hf_backbone_features has freed its only backbone before this head-only path.
    signals = score_cases(
        StoredFeatures(cases, features), tokenizer, config, cases, args
    )
    feature_errors = [
        error_metrics(left, right) for left, right in zip(features, captured)
    ]
    decode_errors = []
    lengths = []
    for case, left, right in zip(cases, features, captured):
        first_decode = case["generation_prompt_length"]
        require(first_decode < left.shape[0], "No actual decode rows in saved sequence")
        decode_errors.append(error_metrics(left[first_decode:], right[first_decode:]))
        lengths.append(
            {
                "generation_prompt_tokens": first_decode,
                "reader_prompt_tokens": len(case["prompt"]),
                "total_tokens": len(case["prompt"]) + len(case["completion"]),
                "feature_rows": left.shape[0],
                "decode_rows": left.shape[0] - first_decode,
            }
        )
    signal_errors = signal_error_metrics(signals, vllm["generated_signals"])
    checks = {
        **parity_checks(feature_errors, signal_errors),
        "decode_features": all(
            error["relative_l2"] <= TOLERANCES["feature_relative_l2"]
            for error in decode_errors
        ),
        "prior_vllm_checks": all(vllm["checks"].values()),
        "generation_capture_without_repair": vllm["generation_query_repairs"] == 0,
    }
    passed = all(checks.values())
    write_json(
        args.output / "generated_parity.json",
        {
            **report_identity(args),
            "phase": "hf-generated",
            "status": "passed" if passed else "failed",
            "overall_status": "passed" if passed else "failed",
            # Short generated cases only; the noise-floor fallback is not used here.
            "pass_criterion": "fixed tolerances only",
            "checks": checks,
            "feature_errors": feature_errors,
            "decode_feature_errors": decode_errors,
            "signal_errors": signal_errors,
            "hf_signals": signals,
            "vllm_signals": vllm["generated_signals"],
            "lengths": lengths,
            "head_only_load_guard_passed": True,
            "input_artifacts": artifact_hashes(args, inputs),
            "output_artifacts": artifact_hashes(args, ("hf_generated_features.pt",)),
            "feature_seconds": timings,
            "phase_seconds": time.perf_counter() - start,
        },
    )
    require(
        passed,
        f"Generated parity failed: {[key for key, ok in checks.items() if not ok]}; see generated_parity.json",
    )
    print("VLLM_GENERATED_PARITY_AND_ALL_CHECKS_OK", flush=True)


def invalidate_phase_artifacts(args):
    """An upstream rerun must never leave a stale downstream success report."""
    generated = (
        "generated_parity.json",
        "hf_generated_features.pt",
        "hf-generated_failure.json",
    )
    vllm = ("vllm.json",) + VLLM_ARTIFACTS + ("vllm_failure.json",)
    names = generated
    if args.phase in ("hf", "vllm"):
        names += vllm
    if args.phase == "hf":
        names += ("hf.json",) + HF_ARTIFACTS + ("hf_failure.json",)
    for name in names:
        (args.output / name).unlink(missing_ok=True)


def preflight(args):
    """Everything that can fail before a phase starts; runs after invalidation."""
    args.output.mkdir(parents=True, exist_ok=True)
    require(
        args.max_num_batched_tokens > 0 and args.max_num_seqs > 0,
        "vLLM scheduling limits must be positive integers",
    )
    args.model = args.model.expanduser().resolve()
    require(
        args.model.is_dir() and args.model.name == BACKBONE_REVISION,
        "--model must be an existing local snapshot ending in the exact pinned revision",
    )
    require(
        torch.cuda.is_available(),
        "This opt-in helper requires a CUDA GPU; no CPU fallback",
    )
    _, record = checkpoint_manifest(DEFAULT_CHECKPOINT, 0, str(args.cache))
    args.checksum = record["sha256"]
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("hf", "vllm", "hf-generated"),
        help="Run hf, then vllm, then hf-generated, each in a fresh process",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help=f"Local Qwen3-1.7B snapshot directory ending in {BACKBONE_REVISION}",
    )
    parser.add_argument(
        "--probe-cache",
        "--cache",
        dest="cache",
        required=True,
        type=Path,
        help="Cache for the pinned BELT seed-0 probe package",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Shared directory for HF reference and vLLM report artifacts",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=8192,
        help="vLLM scheduling token limit; chunked prefill is always enabled (default: 8192)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=8,
        help="Maximum concurrently scheduled vLLM sequences (default: 8)",
    )
    parser.add_argument(
        "--disable-prefix-caching",
        action="store_true",
        help="Disable vLLM APC; the provider's exact-token feature cache remains enabled",
    )
    args = parser.parse_args(argv)
    # First, before any check that can fail: a rerun of this phase must never
    # leave an earlier run's success report (or its downstream reports) behind.
    invalidate_phase_artifacts(args)
    try:
        preflight(args)
        {"hf": hf_phase, "vllm": vllm_phase, "hf-generated": hf_generated_phase}[
            args.phase
        ](args)
    except Exception as error:
        if args.output.is_dir():
            write_json(
                args.output / f"{args.phase}_failure.json",
                {
                    "schema": SCHEMA,
                    "phase": args.phase,
                    "software_test_only": True,
                    "overall_status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise


if __name__ == "__main__":
    main()
