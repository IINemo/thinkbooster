from types import SimpleNamespace

import pytest

from thinkbooster.belt import BeltSignal
from thinkbooster.scorers.step_scorer_belt import StepScorerBELT


class Runtime:
    checkpoint_sha256 = "a" * 64

    def __init__(self):
        self.calls = []

    def score_prefix(self, chat, steps):
        self.calls.append((chat, steps))
        return BeltSignal(
            dict(q=0.2 * len(steps), v=0.6, r=0.5, stability=0.7),
            (1, 2),
            len(steps),
            "a" * 64,
        )


def test_candidate_scoring_includes_history_and_preserves_beliefs():
    runtime = Runtime()
    scorer = StepScorerBELT(runtime=runtime, target="stability")
    chat = [{"role": "user", "content": "q"}]
    scores = scorer.score_candidates_detailed(
        chat, ["candidate a", "candidate b"], trajectory=["history"]
    )
    assert [c[1] for c in runtime.calls] == [
        ["history", "candidate a"],
        ["history", "candidate b"],
    ]
    assert [r.get_score() for r in scores] == [0.7, 0.7]
    assert scores[0].metadata["beliefs"]["q"] == 0.4
    assert scores[0].metadata["checkpoint_sha256"] == "a" * 64


def test_trajectory_scoring_uses_nested_prefixes():
    runtime = Runtime()
    scorer = StepScorerBELT(runtime=runtime)
    assert scorer.score_trajectory([], ["a", "b", "c"]) == pytest.approx(
        [0.2, 0.4, 0.6]
    )
    assert [c[1] for c in runtime.calls] == [["a"], ["a", "b"], ["a", "b", "c"]]


def test_empty_inputs_do_not_load_a_model():
    scorer = StepScorerBELT()
    assert scorer.score_candidates([], []) == []
    assert scorer.score_trajectory([], []) == []
    assert scorer._runtime is None


def test_trajectory_batch_mismatch_fails():
    with pytest.raises(ValueError):
        StepScorerBELT(runtime=SimpleNamespace()).score_trajectories_batch([[]], [])


def test_checkpoint_identity_uses_manifest_without_loading_weights(
    tmp_path, monkeypatch
):
    import json

    from thinkbooster.belt.runtime import (
        BACKBONE,
        BACKBONE_REVISION,
        CHECKPOINT_REVISIONS,
        DEFAULT_CHECKPOINT,
    )

    path = tmp_path / "CHECKPOINTS.json"
    path.write_text(
        json.dumps(
            {
                "backbone": BACKBONE,
                "revision": BACKBONE_REVISION,
                "checkpoints": [
                    {"seed": 0, "sha256": "a" * 64},
                    {"seed": 1, "sha256": "b" * 64},
                ],
            }
        )
    )
    calls = []

    def fetch(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fetch)
    scorer = StepScorerBELT(seed=1, cache_dir=str(tmp_path))
    assert scorer.checkpoint_sha256 == "b" * 64
    assert scorer._runtime is None
    assert calls == [
        (
            DEFAULT_CHECKPOINT,
            "CHECKPOINTS.json",
            {
                "revision": CHECKPOINT_REVISIONS[DEFAULT_CHECKPOINT],
                "cache_dir": str(tmp_path),
            },
        )
    ]


def test_injected_runtime_identity_needs_no_network(monkeypatch):
    def no_download(*args, **kwargs):
        raise AssertionError("The injected runtime already identifies its weights")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", no_download)
    assert StepScorerBELT(runtime=Runtime()).checkpoint_sha256 == "a" * 64


def test_provider_binding_is_lazy_and_cannot_replace_loaded_runtime(monkeypatch):
    from types import SimpleNamespace

    provider, tokenizer, config = object(), object(), object()
    scorer = StepScorerBELT()
    scorer.bind_feature_provider(provider, tokenizer, config)
    assert scorer._runtime is None
    calls = []
    runtime = SimpleNamespace(metadata={"scoring_backend": "vllm"})
    monkeypatch.setattr(
        "thinkbooster.scorers.step_scorer_belt.BeltRuntime.from_pretrained",
        lambda **kwargs: calls.append(kwargs) or runtime,
    )
    assert scorer.runtime is runtime
    assert calls[0]["feature_provider"] is provider
    assert calls[0]["tokenizer"] is tokenizer and calls[0]["base_config"] is config
    with pytest.raises(RuntimeError, match="loaded"):
        scorer.bind_feature_provider(object(), tokenizer)
