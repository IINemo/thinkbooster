import json
import math
from dataclasses import replace

import pytest

from thinkbooster.belt import BeltSignal, BeltStoppingPolicy

SHA = "a" * 64


def signal(hidden=(2.0, 4.0), prefix_tokens=6000):
    return BeltSignal(
        dict(q=0.8, v=0.4, r=0.2, stability=0.6), hidden, prefix_tokens, SHA
    )


def policy(**kwargs):
    values = dict(
        checkpoint_sha256=SHA,
        feature_space="H",
        weights=(3.0, -2.0),
        intercept=0.5,
        means=(1.0, 2.0),
        stds=(1.0, 2.0),
        threshold=1.5,
    )
    values.update(kwargs)
    return BeltStoppingPolicy(**values)


def test_standardized_linear_score_and_threshold_tie():
    fitted = policy()
    assert fitted.decision_logit(signal()) == 1.5
    assert fitted.should_stop(signal())
    assert not policy(threshold=1.50001).should_stop(signal())


def test_belief_logit_order_and_normalized_length():
    fitted = policy(
        feature_space="B",
        weights=(1, 2, 3, 4, 5),
        means=(0, 0, 0, 0, 0),
        stds=(1, 1, 1, 1, 1),
        intercept=-2,
    )
    expected = (
        -2
        + math.log(4)
        + 2 * math.log(2 / 3)
        + 3 * math.log(1 / 4)
        + 4 * math.log(1.5)
        + 2.5
    )
    assert fitted.decision_logit(signal()) == pytest.approx(expected)


def test_belief_probability_clipping():
    observed = replace(signal(), beliefs=dict(q=0, v=1, r=0.5, stability=0.5))
    fitted = policy(
        feature_space="B",
        weights=(1, 1, 0, 0),
        means=(0, 0, 0, 0),
        stds=(1, 1, 1, 1),
        intercept=0,
    )
    assert fitted.decision_logit(observed) == pytest.approx(0, abs=1e-10)


def test_historical_fp16_feature_storage_is_explicit():
    fitted = policy(weights=(1, 0), means=(0, 0), stds=(1, 1), intercept=0)
    observed = signal(hidden=(1.0003, 2.0))
    assert fitted.decision_logit(observed) == 1.0003
    assert replace(fitted, hidden_dtype="float16").decision_logit(observed) == 1.0


def test_fp16_feature_overflow_is_rejected():
    with pytest.raises(ValueError, match="float16"):
        policy(hidden_dtype="float16").decision_logit(signal(hidden=(1e10, 0)))


@pytest.mark.parametrize(
    "change",
    [
        dict(stds=(0, 1)),
        dict(stds=(-1, 1)),
        dict(weights=(1,)),
        dict(feature_space="unknown"),
        dict(threshold=float("nan")),
        dict(length_scale=0),
        dict(logit_clip=0.5),
        dict(checkpoint_sha256="unidentified"),
    ],
)
def test_invalid_policy_is_rejected(change):
    with pytest.raises(ValueError):
        policy(**change)


def test_checkpoint_and_dimension_mismatches_are_rejected():
    with pytest.raises(ValueError, match="SHA256"):
        policy().decision_logit(replace(signal(), checkpoint_sha256="b" * 64))
    with pytest.raises(ValueError, match="width"):
        policy().decision_logit(signal(hidden=(1,)))


def test_policy_file_roundtrip_requires_explicit_threshold(tmp_path):
    path = tmp_path / "policy.json"
    data = dict(vars(policy()), schema="thinkbooster-belt-policy/v1")
    path.write_text(json.dumps(data))
    assert BeltStoppingPolicy.from_file(path) == policy()
    del data["threshold"]
    path.write_text(json.dumps(data))
    with pytest.raises(TypeError):
        BeltStoppingPolicy.from_file(path)


@pytest.mark.parametrize("belief", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_signal_is_rejected(belief):
    with pytest.raises(ValueError):
        replace(signal(), beliefs=dict(q=belief, v=0.5, r=0.5, stability=0.5))
