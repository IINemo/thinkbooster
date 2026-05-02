"""Unit tests for scripts/utils/hook_hs_extension.HookHiddenStatesExtension.

The extension is loaded into a vLLM Worker via worker_extension_cls. We
test it without spinning up vLLM by binding its methods to a mock worker
that emulates model_runner.input_batch and model_runner.query_start_loc,
then driving real torch.nn.Module hooks.
"""

from __future__ import annotations

import os
import pickle
import sys
import types
from typing import Any, List

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

# vllm.distributed is referenced inside the hook closure but only used to
# skip non-rank-0 captures. Use real vllm if it's installed (CI has it via
# the project's dependencies and lm-polygraph needs it for the contract
# integration tests). If vllm is missing, install a minimal stub so the
# basic mock tests still run on a stripped-down dev env.
try:  # pragma: no cover — environment-dependent
    import vllm  # noqa: F401
    import vllm.distributed  # noqa: F401
except ImportError:
    fake_vllm = types.ModuleType("vllm")
    fake_dist = types.ModuleType("vllm.distributed")
    fake_dist.get_tensor_model_parallel_rank = lambda: 0
    sys.modules["vllm"] = fake_vllm
    sys.modules["vllm.distributed"] = fake_dist

from utils.hook_hs_extension import HookHiddenStatesExtension  # noqa: E402


# --------------------------------------------------------------------------- #
# Test scaffolding                                                            #
# --------------------------------------------------------------------------- #


class _Layer(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _BaseInner(torch.nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer() for _ in range(n_layers)])


class _Outer(torch.nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.model = _BaseInner(n_layers)


def _make_worker(req_ids: List[str], offsets: List[int], n_layers: int = 32):
    outer = _Outer(n_layers)
    worker = type("W", (), {})()
    worker.model_runner = type(
        "MR",
        (),
        {
            "input_batch": type("IB", (), {"req_ids": list(req_ids)})(),
            "query_start_loc": type(
                "QSL", (), {"cpu": torch.tensor(offsets, dtype=torch.int64)}
            )(),
            "model": outer,
        },
    )()
    for name in (
        "_get_base_model",
        "_get_request_segmentation",
        "_track_prefill_chunk",
        "_setup_hidden_states_capture",
        "_reset_capture",
        "_get_captured_states",
        "_get_capture_metadata",
    ):
        setattr(worker, name, types.MethodType(getattr(HookHiddenStatesExtension, name), worker))
    return worker, outer


def _set_offsets(worker: Any, offsets: List[int]) -> None:
    worker.model_runner.query_start_loc = type(
        "QSL", (), {"cpu": torch.tensor(offsets, dtype=torch.int64)}
    )()


def _set_req_ids(worker: Any, req_ids: List[str]) -> None:
    worker.model_runner.input_batch = type("IB", (), {"req_ids": list(req_ids)})()


# --------------------------------------------------------------------------- #
# Per-request attribution                                                     #
# --------------------------------------------------------------------------- #


def test_per_request_attribution_shapes_and_values():
    """Each request's slice of the flat batch tensor is attributed correctly."""
    worker, outer = _make_worker(
        req_ids=["r0", "r1", "r2"],
        offsets=[0, 5, 8, 11],  # r0: 5 tokens, r1: 3, r2: 3
    )
    worker._setup_hidden_states_capture([1, 3])

    # Prefill step
    hs = torch.cat(
        [
            torch.ones(5, 8) * 1.0,
            torch.ones(3, 8) * 2.0,
            torch.ones(3, 8) * 3.0,
        ]
    )
    outer.model.layers[1](hs)
    outer.model.layers[3](hs)

    # Decode step (1 token per request)
    _set_offsets(worker, [0, 1, 2, 3])
    hs_d = torch.cat(
        [
            torch.ones(1, 8) * 1.5,
            torch.ones(1, 8) * 2.5,
            torch.ones(1, 8) * 3.5,
        ]
    )
    outer.model.layers[1](hs_d)
    outer.model.layers[3](hs_d)

    captured = worker._get_captured_states()

    for lid in (1, 3):
        assert set(captured[lid]) == {"r0", "r1", "r2"}
        a0 = pickle.loads(captured[lid]["r0"])
        a1 = pickle.loads(captured[lid]["r1"])
        a2 = pickle.loads(captured[lid]["r2"])
        assert a0.shape == (6, 8)
        assert a1.shape == (4, 8)
        assert a2.shape == (4, 8)
        assert (a0[:5] == 1.0).all() and a0[5, 0] == pytest.approx(1.5)
        assert (a1[:3] == 2.0).all() and a1[3, 0] == pytest.approx(2.5)
        assert (a2[:3] == 3.0).all() and a2[3, 0] == pytest.approx(3.5)


# --------------------------------------------------------------------------- #
# Regression: capture metadata must NOT scale with len(hs_layer_ids)          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "hooked_layers",
    [[10], [2, 10], [2, 10, 20], [0, 5, 14, 27]],
)
def test_capture_metadata_independent_of_layer_count(hooked_layers):
    """Metadata bookkeeping runs once per step regardless of hooked-layer count.

    Regression for the multiplier bug: prior code ran
    ``_track_prefill_chunk`` and ``total_computed += chunk`` inside every
    layer's hook closure, multiplying both counters by len(hs_layer_ids).
    Anything that consumes ``prefill_tokens`` (e.g. lm-polygraph's
    ``_fill_prefix_gaps``) then produced incorrect prefix fills.
    """
    worker, outer = _make_worker(req_ids=["r0"], offsets=[0, 5])
    worker._setup_hidden_states_capture(hooked_layers)

    hs = torch.ones(5, 8)
    for lid in hooked_layers:
        outer.model.layers[lid](hs)

    meta = worker._get_capture_metadata()
    assert meta["r0"]["prefill_tokens"] == 5, meta
    assert meta["r0"]["total_computed"] == 5, meta


def test_capture_metadata_multistep_decode():
    """Across prefill + multiple decode steps, counters add up correctly."""
    worker, outer = _make_worker(req_ids=["r0", "r1"], offsets=[0, 5, 8])
    worker._setup_hidden_states_capture([2, 10, 20])  # 3 layers

    # Prefill: r0 gets 5, r1 gets 3
    hs = torch.ones(8, 8)
    for lid in (2, 10, 20):
        outer.model.layers[lid](hs)

    # Decode 1: 1 token per request
    _set_offsets(worker, [0, 1, 2])
    hs_d = torch.ones(2, 8)
    for lid in (2, 10, 20):
        outer.model.layers[lid](hs_d)

    # Decode 2
    for lid in (2, 10, 20):
        outer.model.layers[lid](hs_d)

    meta = worker._get_capture_metadata()
    assert meta["r0"]["prefill_tokens"] == 5
    assert meta["r0"]["total_computed"] == 5 + 1 + 1
    assert meta["r1"]["prefill_tokens"] == 3
    assert meta["r1"]["total_computed"] == 3 + 1 + 1


# --------------------------------------------------------------------------- #
# reset and edge cases                                                        #
# --------------------------------------------------------------------------- #


def test_reset_capture_clears_all_state():
    worker, outer = _make_worker(req_ids=["r0"], offsets=[0, 5])
    worker._setup_hidden_states_capture([5])
    outer.model.layers[5](torch.ones(5, 8))

    worker._reset_capture()
    assert worker._hs_captured == {}
    assert worker._hs_req_meta == {}
    assert worker._hs_prefill_tokens == {}


def test_get_captured_states_returns_nested_dict():
    """Contract: {layer_id: {req_id: pickle_bytes}} — not flat bytes.

    lm-polygraph's _raw_generate (PR 453+) iterates this nested form.
    """
    worker, outer = _make_worker(req_ids=["r0", "r1"], offsets=[0, 3, 5])
    worker._setup_hidden_states_capture([7])
    outer.model.layers[7](torch.ones(5, 8))

    captured = worker._get_captured_states()
    assert isinstance(captured, dict)
    assert 7 in captured
    assert isinstance(captured[7], dict)
    assert {"r0", "r1"} == set(captured[7])
    assert isinstance(captured[7]["r0"], (bytes, bytearray))
    arr = pickle.loads(captured[7]["r0"])
    assert arr.shape == (3, 8)


# --------------------------------------------------------------------------- #
# Hook plumbing — TP, tuple/3D outputs, fallback, re-registration             #
# --------------------------------------------------------------------------- #


def test_non_rank0_tp_does_not_capture(monkeypatch):
    """Only TP rank 0 captures HS — other ranks must early-return.

    All TP ranks see identical hidden states; capturing on more than one
    would waste memory and (more critically) double the metadata if anyone
    naively merges across ranks.
    """
    import vllm.distributed as fake

    monkeypatch.setattr(fake, "get_tensor_model_parallel_rank", lambda: 1)

    worker, outer = _make_worker(req_ids=["r0"], offsets=[0, 5])
    worker._setup_hidden_states_capture([5])
    outer.model.layers[5](torch.ones(5, 8))

    assert worker._hs_captured == {}
    assert worker._hs_req_meta == {}
    assert worker._hs_prefill_tokens == {}


def test_3d_output_is_flattened():
    """vLLM may return [batch, seq, hidden] for some paths — must flatten."""
    worker, outer = _make_worker(req_ids=["r0"], offsets=[0, 4])
    worker._setup_hidden_states_capture([3])

    # 3D: [batch=1, seq=4, hidden=8]
    outer.model.layers[3](torch.ones(1, 4, 8))

    captured = worker._get_captured_states()
    arr = pickle.loads(captured[3]["r0"])
    assert arr.shape == (4, 8)


def test_tuple_output_is_unwrapped():
    """Some layers return (hidden_states, ...) — hook must take element 0."""

    class TupleLayer(torch.nn.Module):
        def forward(self, x):
            return (x, None)  # (hs, residual)

    outer = _Outer(8)
    outer.model.layers[2] = TupleLayer()
    worker = type("W", (), {})()
    worker.model_runner = type(
        "MR",
        (),
        {
            "input_batch": type("IB", (), {"req_ids": ["r0"]})(),
            "query_start_loc": type(
                "QSL", (), {"cpu": torch.tensor([0, 4], dtype=torch.int64)}
            )(),
            "model": outer,
        },
    )()
    for name in (
        "_get_base_model",
        "_get_request_segmentation",
        "_track_prefill_chunk",
        "_setup_hidden_states_capture",
        "_reset_capture",
        "_get_captured_states",
        "_get_capture_metadata",
    ):
        setattr(worker, name, types.MethodType(getattr(HookHiddenStatesExtension, name), worker))

    worker._setup_hidden_states_capture([2])
    outer.model.layers[2](torch.ones(4, 8))

    captured = worker._get_captured_states()
    assert pickle.loads(captured[2]["r0"]).shape == (4, 8)


def test_unknown_fallback_when_no_segmentation(caplog):
    """If query_start_loc is missing, capture under '__unknown__' instead of crashing."""
    worker, outer = _make_worker(req_ids=[], offsets=[0])  # empty batch
    worker._setup_hidden_states_capture([5])

    import logging

    with caplog.at_level(logging.WARNING):
        outer.model.layers[5](torch.ones(3, 8))

    captured = worker._get_captured_states()
    # Empty req_ids → no segmentation → __unknown__
    assert "__unknown__" in captured[5]


def test_setup_twice_removes_stale_hooks():
    """Calling _setup_hidden_states_capture again must remove old hooks."""
    worker, outer = _make_worker(req_ids=["r0"], offsets=[0, 4])

    worker._setup_hidden_states_capture([1, 2, 3])
    assert len(worker._hs_hook_handles) == 3

    # Re-setup with different layers
    worker._setup_hidden_states_capture([5])
    assert len(worker._hs_hook_handles) == 1

    outer.model.layers[5](torch.ones(4, 8))
    # Old layers must NOT have captured anything
    outer.model.layers[1](torch.ones(4, 8))
    outer.model.layers[2](torch.ones(4, 8))
    outer.model.layers[3](torch.ones(4, 8))

    captured = worker._get_captured_states()
    assert set(captured.keys()) == {5}, captured.keys()


def test_setup_with_empty_layers_registers_no_hooks():
    """Edge case: empty hs_layer_ids must not crash and not hook anything."""
    worker, outer = _make_worker(req_ids=["r0"], offsets=[0, 4])
    worker._setup_hidden_states_capture([])
    assert len(worker._hs_hook_handles) == 0

    outer.model.layers[5](torch.ones(4, 8))
    assert worker._get_captured_states() == {}


def test_multiple_generate_cycles():
    """_get_captured_states clears _hs_captured between calls.

    A new generate() must not see stale data from the prior call.
    """
    worker, outer = _make_worker(req_ids=["r0"], offsets=[0, 5])
    worker._setup_hidden_states_capture([10])

    outer.model.layers[10](torch.ones(5, 8) * 1.0)
    first = worker._get_captured_states()
    arr1 = pickle.loads(first[10]["r0"])
    assert arr1.shape == (5, 8) and (arr1 == 1.0).all()

    # Second cycle: different req_ids, different values
    _set_req_ids(worker, ["r1"])
    _set_offsets(worker, [0, 3])
    worker._reset_capture()
    outer.model.layers[10](torch.ones(3, 8) * 2.0)
    second = worker._get_captured_states()
    arr2 = pickle.loads(second[10]["r1"])
    assert arr2.shape == (3, 8) and (arr2 == 2.0).all()
    # Old req_id must NOT bleed through
    assert "r0" not in second[10]


# --------------------------------------------------------------------------- #
# Contract integration with lm-polygraph's _fill_prefix_gaps                  #
# --------------------------------------------------------------------------- #


def _drive_apc_scenario(hooked_layers):
    """Drive a 3-request scenario that exercises both fill phases.

      req=0 — full prefill (donor)                                        len 10
      req=1 — same prompt as req=0 but APC reused 7 of 10 tokens          len 3
      req=2 — different prompt, shares first 5 tokens with req=0          len 4
              (APC reused those 5, computed remaining 4 fresh)
    Decode adds 2 tokens per request.
    """
    worker, outer = _make_worker(
        req_ids=["0", "1", "2"], offsets=[0, 10, 13, 17]
    )
    worker._setup_hidden_states_capture(hooked_layers)

    # Prefill: req=0 has 10 (full), req=1 has 3 (APC reused 7), req=2 has 4 (APC reused 5)
    parts = [
        torch.ones(10, 8) * 1.0,  # req=0  prompt tokens 0..9
        torch.ones(3, 8) * 2.0,  # req=1  last 3 prompt tokens (8..10 of identical prompt)
        torch.ones(4, 8) * 3.0,  # req=2  prompt tokens 5..8 (after shared prefix)
    ]
    hs_prefill = torch.cat(parts, dim=0)
    for lid in hooked_layers:
        outer.model.layers[lid](hs_prefill)

    # Decode 1: 1 token per request
    _set_offsets(worker, [0, 1, 2, 3])
    decode = torch.cat([torch.ones(1, 8) * 1.5, torch.ones(1, 8) * 2.5, torch.ones(1, 8) * 3.5])
    for lid in hooked_layers:
        outer.model.layers[lid](decode)

    # Decode 2
    for lid in hooked_layers:
        outer.model.layers[lid](decode)

    return worker, outer


def test_contract_with_lm_polygraph_fill_prefix_gaps_single_layer():
    """End-to-end-ish: drive the hook, then run the canonical lm-polygraph
    helper. Confirms shapes + content invariants without GPU.
    """
    pytest.importorskip("lm_polygraph.utils.vllm_with_uncertainty")
    from lm_polygraph.utils.vllm_with_uncertainty import _fill_prefix_gaps

    worker, _ = _drive_apc_scenario(hooked_layers=[10])
    captured = worker._get_captured_states()
    meta = worker._get_capture_metadata()

    arrs = {rid: pickle.loads(blob) for rid, blob in captured[10].items()}

    # Identical prompts for req=0 and req=1; req=2 differs but shares 5-token prefix
    prompt_groups = {"prompt_A": ["0", "1"], "prompt_B": ["2"]}
    prompt_tokens = {
        "0": list(range(10)),
        "1": list(range(10)),  # same tokens as req=0
        "2": list(range(5)) + [100, 101, 102, 103],  # shared first 5 tokens
    }

    filled = _fill_prefix_gaps(arrs, meta, prompt_groups, prompt_tokens=prompt_tokens)

    # Donor unchanged
    assert filled["0"].shape == (12, 8)  # 10 prompt + 2 decode
    # Within-group: req=1 should have its missing 7 prompt tokens filled from donor
    assert filled["1"].shape == (12, 8), filled["1"].shape
    # First 7 rows are donor's prompt tokens (value 1.0); rows 7..9 are req=1's own
    # 3 prefill tokens (value 2.0); rows 10..11 are req=1's 2 decode tokens (value 2.5)
    assert (filled["1"][:7] == 1.0).all(), "phase 1 prefix copy"
    assert (filled["1"][7:10] == 2.0).all(), "phase 1 must not corrupt own prefill"
    assert (filled["1"][10:] == 2.5).all(), "phase 1 must not corrupt own decode"

    # Cross-group: req=2 missing 5 prompt tokens; LCP with donor (req=0) is 5
    assert filled["2"].shape == (11, 8), filled["2"].shape  # 5 from donor + 4 own + 2 decode
    assert (filled["2"][:5] == 1.0).all(), "phase 2 LCP-fill prefix"
    assert (filled["2"][5:9] == 3.0).all(), "phase 2 must not corrupt own prefill"
    assert (filled["2"][9:] == 3.5).all(), "phase 2 must not corrupt own decode"


@pytest.mark.parametrize("hooked_layers", [[1], [10], [2, 10, 20], [0, 5, 14, 27]])
def test_contract_with_lm_polygraph_fill_prefix_gaps_multilayer(hooked_layers):
    """Same scenario, varying layer count: filled sizes must remain identical.

    This is THE test that catches a regression of the multiplier bug — if
    bookkeeping is double-counted, _fill_prefix_gaps will produce different
    sizes for different layer counts.
    """
    pytest.importorskip("lm_polygraph.utils.vllm_with_uncertainty")
    from lm_polygraph.utils.vllm_with_uncertainty import _fill_prefix_gaps

    worker, _ = _drive_apc_scenario(hooked_layers=hooked_layers)
    captured = worker._get_captured_states()
    meta = worker._get_capture_metadata()

    sample_layer = hooked_layers[0]
    arrs = {rid: pickle.loads(blob) for rid, blob in captured[sample_layer].items()}
    prompt_groups = {"prompt_A": ["0", "1"], "prompt_B": ["2"]}
    prompt_tokens = {
        "0": list(range(10)),
        "1": list(range(10)),
        "2": list(range(5)) + [100, 101, 102, 103],
    }
    filled = _fill_prefix_gaps(arrs, meta, prompt_groups, prompt_tokens=prompt_tokens)

    # Sizes are independent of len(hooked_layers)
    assert filled["0"].shape == (12, 8), (hooked_layers, filled["0"].shape)
    assert filled["1"].shape == (12, 8), (hooked_layers, filled["1"].shape)
    assert filled["2"].shape == (11, 8), (hooked_layers, filled["2"].shape)


# --------------------------------------------------------------------------- #
# Strategy integration: reset_hs_step_cache must be called once per loop      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "module_path,class_name",
    [
        ("thinkbooster.strategies.strategy_beam_search", "StrategyBeamSearch"),
        ("thinkbooster.strategies.strategy_online_best_of_n", "StrategyOnlineBestOfN"),
        ("thinkbooster.strategies.strategy_extended_thinking", "StrategyExtendedThinking"),
        ("thinkbooster.strategies.strategy_uncertainty_cot", "StrategyUncertaintyCoT"),
        ("thinkbooster.strategies.adaptive_scaling_best_of_n", "AdaptiveScalingBestOfN"),
    ],
)
def test_strategy_calls_reset_hs_step_cache_when_available(module_path, class_name):
    """Each multi-step strategy must call reset_hs_step_cache once at start.

    The strategies guard the call with hasattr(model, "reset_hs_step_cache"),
    so this test verifies the hasattr-positive branch works. Source-level
    check (no execution): we just confirm the call is present and gated.
    """
    import importlib
    import inspect

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    src = inspect.getsource(cls)
    # The exact 6-line block we added
    assert "reset_hs_step_cache" in src, f"{class_name} missing reset_hs_step_cache call"
    assert 'hasattr(model, "reset_hs_step_cache")' in src, (
        f"{class_name} call must be guarded by hasattr (back-compat)"
    )
