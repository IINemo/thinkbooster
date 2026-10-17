from types import SimpleNamespace

import pytest

from thinkbooster.belt import BeltSignal
from thinkbooster.scorers.step_scorer_belt import StepScorerBELT


class Runtime:
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
