"""Primitive-only residual payloads for vLLM's untyped utility-RPC results.

vLLM 0.12 does not reconstruct nested NumPy arrays in UtilityResult unless
insecure serialization is enabled. Use an explicit endian/shape/bytes contract
instead of depending on that optional type metadata or its internal tuple form.
"""

import numpy as np

FEATURE_WIRE_SCHEMA = "thinkbooster-belt-f32/v1"
FEATURE_WIDTH = 2048
MAX_CAPTURE_TOKENS = 131072


def encode_features(features):
    """Encode finite float32 rows as a C-order, little-endian byte payload."""
    if (
        not isinstance(features, np.ndarray)
        or features.dtype != np.float32
        or features.ndim != 2
        or features.shape[1] != FEATURE_WIDTH
        or not 1 <= features.shape[0] <= MAX_CAPTURE_TOKENS
    ):
        raise RuntimeError("BELT feature wire requires bounded float32 [N,2048] rows")
    if not np.isfinite(features).all():
        raise RuntimeError("BELT feature wire contains nonfinite values")
    rows = np.ascontiguousarray(features, dtype=np.dtype("<f4"))
    return {
        "schema": FEATURE_WIRE_SCHEMA,
        "dtype": "<f4",
        "shape": list(rows.shape),
        "data": rows.tobytes(order="C"),
    }


def decode_features(payload, max_rows=MAX_CAPTURE_TOKENS):
    """Validate before allocating; return independent, native float32 CPU rows."""
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "dtype",
        "shape",
        "data",
    }:
        raise RuntimeError("BELT feature wire requires a versioned byte envelope")
    shape, data = payload["shape"], payload["data"]
    if (
        payload["schema"] != FEATURE_WIRE_SCHEMA
        or payload["dtype"] != "<f4"
        or not isinstance(shape, list)
        or len(shape) != 2
        or any(type(value) is not int for value in shape)
        or not 1 <= shape[0] <= min(max_rows, MAX_CAPTURE_TOKENS)
        or shape[1] != FEATURE_WIDTH
        or type(data) is not bytes
        or len(data) != shape[0] * FEATURE_WIDTH * 4
    ):
        raise RuntimeError(
            "BELT feature wire has invalid schema/dtype/shape/byte length"
        )
    rows = np.frombuffer(data, dtype=np.dtype("<f4")).reshape(shape)
    if not np.isfinite(rows).all():
        raise RuntimeError("BELT feature wire contains nonfinite values")
    return rows.astype(np.float32, order="C", copy=True)
