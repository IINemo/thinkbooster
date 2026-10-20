"""CPU checks for the GPU helper's ownership/error handling, without an engine."""

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def owned_engine(monkeypatch):
    helper = runpy.run_path(str(Path(__file__).with_name("validate_vllm_gpu.py")))
    calls = []
    raw = SimpleNamespace(
        llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(shutdown=lambda: calls.append("shutdown"))
        )
    )
    stub = ModuleType("vllm")
    stub.LLM = lambda **kwargs: calls.append(kwargs) or raw
    monkeypatch.setitem(sys.modules, "vllm", stub)
    return helper["owned_vllm_engine"], raw, calls


def test_helper_owns_and_shuts_down_its_engine_on_success(owned_engine):
    context, raw, calls = owned_engine
    with context({"model": "fixture"}) as actual:
        assert actual is raw
        assert calls == [{"model": "fixture"}]
    assert calls == [{"model": "fixture"}, "shutdown"]


def test_helper_shuts_down_after_body_failure(owned_engine):
    context, _, calls = owned_engine
    failure = ValueError("original body failure")
    with pytest.raises(ValueError) as received:
        with context({}):
            raise failure
    assert received.value is failure
    assert calls == [{}, "shutdown"]


@pytest.mark.parametrize("body_fails", [False, True])
def test_helper_does_not_mask_original_exception_when_shutdown_fails(
    owned_engine, body_fails
):
    context, raw, _ = owned_engine
    shutdown_failure = RuntimeError("cleanup failure")
    original = ValueError("original body failure")

    def shutdown():
        raise shutdown_failure

    raw.llm_engine.engine_core.shutdown = shutdown
    with pytest.raises(ValueError if body_fails else RuntimeError) as received:
        with context({}):
            if body_fails:
                raise original
    assert received.value is (original if body_fails else shutdown_failure)


@pytest.fixture
def helper():
    return runpy.run_path(str(Path(__file__).with_name("validate_vllm_gpu.py")))


STALE_PASSED = {
    "hf": ("hf.json", "cases.json", "hf_features.pt", "hf_fp32_features.pt"),
    "vllm": (
        "vllm.json",
        "generated_cases.json",
        "vllm_generated_features.pt",
        "vllm_features.pt",
    ),
    "hf-generated": ("generated_parity.json", "hf_generated_features.pt"),
}
# Each phase invalidates itself and everything downstream of it.
INVALIDATED = {
    "hf": ("hf", "vllm", "hf-generated"),
    "vllm": ("vllm", "hf-generated"),
    "hf-generated": ("hf-generated",),
}


@pytest.mark.parametrize("phase", ["hf", "vllm", "hf-generated"])
@pytest.mark.parametrize("failure", ["missing_model", "no_cuda", "manifest"])
def test_failed_preflight_cannot_leave_stale_passed_report(
    helper, monkeypatch, tmp_path, phase, failure
):
    main = helper["main"]
    output = tmp_path / "out"
    output.mkdir()
    for names in STALE_PASSED.values():
        for name in names:
            (output / name).write_text('{"status": "passed"}\n')
    model = tmp_path / helper["BACKBONE_REVISION"]
    if failure != "missing_model":
        model.mkdir()
    monkeypatch.setattr(
        helper["torch"].cuda, "is_available", lambda: failure != "no_cuda"
    )

    def manifest(*args, **kwargs):
        raise OSError("probe manifest unavailable")

    monkeypatch.setitem(main.__globals__, "checkpoint_manifest", manifest)
    argv = [phase, "--model", str(model), "--cache", str(tmp_path / "cache")]
    with pytest.raises(OSError if failure == "manifest" else RuntimeError):
        main(argv + ["--output", str(output)])

    for owner, names in STALE_PASSED.items():
        for name in names:
            assert (output / name).exists() is (owner not in INVALIDATED[phase])
    report = json.loads((output / f"{phase}_failure.json").read_text())
    assert report["phase"] == phase and report["overall_status"] == "failed"


def test_preflight_failure_into_new_output_directory_records_failure(helper, tmp_path):
    output = tmp_path / "fresh"
    argv = ["vllm", "--model", str(tmp_path / "missing"), "--cache", str(tmp_path)]
    with pytest.raises(RuntimeError, match="pinned revision"):
        helper["main"](argv + ["--output", str(output)])
    assert json.loads((output / "vllm_failure.json").read_text())["error_type"] == (
        "RuntimeError"
    )


def feature_error(relative_l2):
    return {"max_abs": 0.0, "rms": 0.0, "relative_l2": relative_l2, "cosine": 1.0}


def signal_error(hidden, belief, logit):
    return {
        "hidden": feature_error(hidden),
        "max_belief_abs": belief,
        "policy_logit_abs": logit,
    }


def test_noise_aware_checks_record_both_numbers_and_the_criterion(helper):
    checks_for = helper["noise_aware_checks"]
    factor = helper["NOISE_FACTOR"]
    cases = [{"name": "short"}, {"name": "long"}]
    floor = [
        {
            "name": "short",
            "features": feature_error(0.005),
            "signals": signal_error(0.005, 0.002, 0.01),
        },
        {
            "name": "long",
            "features": feature_error(0.02),
            "signals": signal_error(0.02, 0.01, 0.01),
        },
    ]
    # short: inside every fixed tolerance. long: features/hidden/beliefs miss the
    # tolerance but stay within factor * floor against fp32; logits pass outright.
    versus_bf16 = (
        [feature_error(0.008), feature_error(0.026)],
        [signal_error(0.008, 0.001, 0.01), signal_error(0.03, 0.015, 0.01)],
    )
    versus_fp32 = (
        [feature_error(0.009), feature_error(0.03)],
        [signal_error(0.009, 0.001, 0.01), signal_error(0.035, 0.012, 0.01)],
    )
    checks, rows, via_floor = checks_for(cases, *versus_bf16, *versus_fp32, floor)
    assert all(checks.values())
    assert via_floor == ["long:features", "long:hidden", "long:beliefs"]
    assert {row[check]["passed_by"] for row in rows[:1] for check in checks} == {
        "tolerance"
    }
    long_features = rows[1]["features"]
    assert long_features == {
        "passed_by": "noise_floor",
        "vllm_vs_hf_bf16": 0.026,
        "tolerance": 0.02,
        "vllm_vs_hf_fp32": 0.03,
        "hf_bf16_vs_hf_fp32": 0.02,
        "noise_bound": factor * 0.02,
        "vllm_fp32_to_floor_ratio": pytest.approx(1.5),
    }
    assert rows[1]["fixture_policy_logits"]["passed_by"] == "tolerance"

    # A real bug: far outside both the tolerance and the noise bound.
    versus_fp32[0][1] = feature_error(factor * 0.02 + 1e-6)
    checks, rows, via_floor = checks_for(cases, *versus_bf16, *versus_fp32, floor)
    assert checks["features"] is False and rows[1]["features"]["passed_by"] == "failed"
    assert "long:features" not in via_floor
    # The strict, tolerance-only view stays available and fails the long case.
    assert helper["parity_checks"](*versus_bf16)["features"] is False


def test_noise_aware_checks_zero_floor_is_strict(helper):
    zero = {
        "name": "a",
        "features": feature_error(0.0),
        "signals": signal_error(0.0, 0.0, 0.0),
    }
    checks, rows, _ = helper["noise_aware_checks"](
        [{"name": "a"}],
        [feature_error(0.03)],
        [signal_error(0.0, 0.0, 0.0)],
        [feature_error(1e-9)],
        [signal_error(0.0, 0.0, 0.0)],
        [zero],
    )
    assert checks["features"] is False
    assert rows[0]["features"]["vllm_fp32_to_floor_ratio"] is None


def test_position_profile_locates_the_worst_rows(helper):
    torch = helper["torch"]
    reference = torch.ones(20, 4)
    actual = reference.clone()
    actual[17] += 0.5
    actual[3] += 0.1
    token_ids = list(range(100, 121))
    profile = helper["position_profile"](reference, actual, token_ids, 5)
    assert profile["rows"] == 20 and profile["prompt_rows"] == 5
    assert len(profile["buckets"]) == helper["PROFILE_BUCKETS"]
    assert [b["start"] for b in profile["buckets"]][:3] == [0, 2, 4]
    assert profile["buckets"][-1]["stop"] == 20
    worst = profile["worst_positions"]
    assert len(worst) == helper["PROFILE_TOP_K"]
    assert worst[0] == {
        "position": 17,
        "token_id": 117,
        "completion_index": 12,
        "relative_l2": pytest.approx(0.5),
    }
    assert worst[1]["position"] == 3 and worst[1]["completion_index"] is None
    assert profile["row_relative_l2"]["max"] == pytest.approx(0.5)
    assert profile["buckets"][8]["max_row_relative_l2"] == pytest.approx(0.5)
    json.dumps(profile, allow_nan=False)
    with pytest.raises(RuntimeError):
        helper["position_profile"](reference, actual, token_ids[:-1], 5)
