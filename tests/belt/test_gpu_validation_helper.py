"""CPU checks for the GPU helper's ownership/error handling, without an engine."""

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
