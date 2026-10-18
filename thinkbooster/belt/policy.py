"""Evaluate fitted linear policies on BELT signals."""

import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

TARGETS = ("q", "v", "r", "stability")


@dataclass(frozen=True)
class BeltSignal:
    beliefs: Mapping[str, float]
    hidden: Tuple[float, ...]
    prefix_tokens: int
    checkpoint_sha256: str

    def __post_init__(self):
        if set(self.beliefs) != set(TARGETS):
            raise ValueError(f"BELT requires beliefs in {TARGETS}")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in self.beliefs.values()):
            raise ValueError("BELT beliefs must be finite probabilities")
        if not self.hidden or not all(math.isfinite(x) for x in self.hidden):
            raise ValueError("BELT hidden representation must be nonempty and finite")
        if type(self.prefix_tokens) is not int or self.prefix_tokens < 1:
            raise ValueError("prefix_tokens must be a positive integer")


@dataclass(frozen=True)
class BeltStoppingPolicy:
    """Standardized linear score; stop when the score reaches threshold.

    H consumes the hidden representation. B consumes clipped belief logits in
    q/v/r/stability order, optionally followed by normalized prefix length.
    Parameters and threshold must come from a compatible policy fit.
    """

    checkpoint_sha256: str
    feature_space: str
    weights: Tuple[float, ...]
    intercept: float
    means: Tuple[float, ...]
    stds: Tuple[float, ...]
    threshold: float
    length_scale: float = 12000.0
    logit_clip: float = 1e-4
    hidden_dtype: str = "float32"

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must identify the fitted probe")
        if self.feature_space not in ("H", "B"):
            raise ValueError("feature_space must be H or B")
        if self.hidden_dtype not in ("float32", "float16"):
            raise ValueError("hidden_dtype must be float32 or float16")
        width = len(self.weights)
        if width == 0 or len(self.means) != width or len(self.stds) != width:
            raise ValueError("weights, means and stds must have equal nonzero lengths")
        if self.feature_space == "B" and width not in (4, 5):
            raise ValueError("B policies require four beliefs, optionally with length")
        values = (
            *self.weights,
            *self.means,
            *self.stds,
            self.intercept,
            self.threshold,
        )
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Policy parameters must be finite")
        if any(x <= 0 for x in self.stds):
            raise ValueError("Policy standard deviations must be positive")
        if not math.isfinite(self.length_scale) or self.length_scale <= 0:
            raise ValueError("length_scale must be positive and finite")
        if not math.isfinite(self.logit_clip) or not 0 < self.logit_clip < 0.5:
            raise ValueError("logit_clip must lie between 0 and 0.5")

    @classmethod
    def from_file(cls, path):
        data = json.loads(Path(path).read_text())
        if data.pop("schema", None) != "thinkbooster-belt-policy/v1":
            raise ValueError("Expected thinkbooster-belt-policy/v1")
        for key in ("weights", "means", "stds"):
            data[key] = tuple(data[key])
        return cls(**data)

    def validate_checkpoint(self, checkpoint_sha256):
        if checkpoint_sha256 != self.checkpoint_sha256:
            raise ValueError(
                "Policy checkpoint SHA256 differs from the selected BELT probe"
            )

    def decision_logit(self, signal: BeltSignal) -> float:
        self.validate_checkpoint(signal.checkpoint_sha256)
        if self.feature_space == "H":
            features = signal.hidden
            if self.hidden_dtype == "float16":
                try:
                    features = tuple(
                        struct.unpack("e", struct.pack("e", x))[0] for x in features
                    )
                except (OverflowError, struct.error) as error:
                    raise ValueError(
                        "Hidden representation cannot be stored as float16"
                    ) from error
        else:
            features = []
            for name in TARGETS:
                p = min(max(signal.beliefs[name], self.logit_clip), 1 - self.logit_clip)
                features.append(math.log(p / (1 - p)))
            if len(self.weights) == 5:
                features.append(signal.prefix_tokens / self.length_scale)
        if len(features) != len(self.weights):
            raise ValueError("Policy width differs from the BELT feature width")
        value = float(self.intercept)
        for x, w, mean, std in zip(features, self.weights, self.means, self.stds):
            value += float(w) * ((float(x) - float(mean)) / float(std))
        if not math.isfinite(value):
            raise ValueError("BELT policy produced a nonfinite score")
        return value

    def should_stop(self, signal: BeltSignal) -> bool:
        return self.decision_logit(signal) >= self.threshold
