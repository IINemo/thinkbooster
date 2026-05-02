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
# skip non-rank-0 captures. Stub it for tests so we don't import vLLM.
fake_dist = types.ModuleType("vllm.distributed")
fake_dist.get_tensor_model_parallel_rank = lambda: 0
sys.modules.setdefault("vllm", types.ModuleType("vllm"))
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
