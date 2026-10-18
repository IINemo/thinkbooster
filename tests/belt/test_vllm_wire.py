"""Strict feature wire contract, including the actual vLLM utility-RPC codec."""

import importlib
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pytest

from thinkbooster.belt.vllm_wire import (
    MAX_CAPTURE_TOKENS,
    decode_features,
    encode_features,
)


def test_wire_is_explicit_little_endian_bytes_and_decode_owns_storage():
    original = np.arange(4 * 2048, dtype=np.float32).reshape(4, 2048)[::2]
    payload = encode_features(original)
    assert payload["dtype"] == "<f4" and payload["shape"] == [2, 2048]
    assert type(payload["data"]) is bytes and len(payload["data"]) == 2 * 2048 * 4
    decoded = decode_features(payload)
    assert decoded.dtype == np.float32 and decoded.flags.owndata
    assert decoded.flags.c_contiguous and decoded.flags.writeable
    np.testing.assert_array_equal(decoded, original)
    original.fill(-1)
    assert decoded[0, 1] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "unsupported"},
        {"dtype": ">f4"},
        {"dtype": "<f8"},
        {"shape": [1, 1024]},
        {"shape": [True, 2048]},
        {"shape": [-1, 2048]},
        {"shape": [MAX_CAPTURE_TOKENS + 1, 2048]},
        {"data": b"broken"},
        {"data": bytearray(2048 * 4)},
        {"extra": True},
    ],
)
def test_wire_rejects_invalid_headers_and_lengths(change):
    payload = encode_features(np.ones((1, 2048), dtype=np.float32))
    payload.update(change)
    with pytest.raises(RuntimeError, match="BELT feature wire"):
        decode_features(payload)


def test_wire_rejects_nonfinite_values_and_remaining_capture_bound():
    rows = np.ones((2, 2048), dtype=np.float32)
    payload = encode_features(rows)
    with pytest.raises(RuntimeError, match="BELT feature wire"):
        decode_features(payload, max_rows=1)
    rows[1, 7] = np.nan
    payload["data"] = rows.tobytes()
    with pytest.raises(RuntimeError, match="nonfinite"):
        decode_features(payload)
    with pytest.raises(RuntimeError, match="nonfinite"):
        encode_features(rows)


@pytest.fixture
def real_vllm_serialization():
    # The local CPU-only development environment deliberately omits vLLM.
    # Run these cases in CI / the installed vLLM environment without a GPU.
    try:
        installed = version("vllm")
    except PackageNotFoundError:
        pytest.skip(
            "Real vLLM package required for utility-RPC serialization regression"
        )
    if installed != "0.12.0":
        pytest.skip("Utility-RPC regression targets the supported vLLM 0.12.0")
    return (
        importlib.import_module("vllm.v1.serial_utils"),
        importlib.import_module("vllm.v1.engine"),
    )


def rpc_roundtrip(serial, engine, reply, threshold):
    message = engine.EngineCoreOutputs(
        utility_output=engine.UtilityOutput(
            call_id=7, result=serial.UtilityResult([reply])
        )
    )
    frames = serial.MsgpackEncoder(size_threshold=threshold).encode(message)
    received = serial.MsgpackDecoder(engine.EngineCoreOutputs).decode(frames)
    return received.utility_output.result.result[0], frames


@pytest.mark.parametrize("insecure", ["0", "1"])
@pytest.mark.parametrize("threshold", [0, 1 << 30])
def test_real_vllm_utility_rpc_preserves_explicit_feature_payload(
    real_vllm_serialization,
    monkeypatch,
    insecure,
    threshold,
):
    serial, engine = real_vllm_serialization
    monkeypatch.setitem(
        serial.envs.__dict__, "VLLM_ALLOW_INSECURE_SERIALIZATION", insecure == "1"
    )
    assert serial.envs.VLLM_ALLOW_INSECURE_SERIALIZATION is (insecure == "1")
    # Actual worker replies are a singleton list wrapped in UtilityResult.
    rows = np.arange(3 * 2048, dtype=np.float32).reshape(3, 2048)
    sent = {"request": {"positions": [3, 4, 5], "features": encode_features(rows)}}
    received, frames = rpc_roundtrip(serial, engine, sent, threshold)
    assert len(frames) == 1  # No custom ndarray hook or auxiliary NumPy frame.
    assert received == sent
    np.testing.assert_array_equal(
        decode_features(received["request"]["features"]), rows
    )


@pytest.mark.parametrize("threshold", [0, 1 << 30])
def test_real_vllm_reproduces_legacy_numpy_utility_result_failure(
    real_vllm_serialization,
    monkeypatch,
    threshold,
):
    serial, engine = real_vllm_serialization
    monkeypatch.setitem(
        serial.envs.__dict__, "VLLM_ALLOW_INSECURE_SERIALIZATION", False
    )
    assert serial.envs.VLLM_ALLOW_INSECURE_SERIALIZATION is False
    rows = np.ones((3, 2048), dtype=np.float32)
    received, _ = rpc_roundtrip(
        serial,
        engine,
        {"request": {"positions": [0, 1, 2], "features": rows}},
        threshold,
    )
    encoded_array = received["request"]["features"]
    assert not isinstance(encoded_array, np.ndarray) and len(encoded_array) == 3
    with pytest.raises(ValueError):
        np.asarray(encoded_array)
