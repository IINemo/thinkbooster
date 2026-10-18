import threading
from types import SimpleNamespace

import pytest

from thinkbooster.belt import BeltSignal, BeltStoppingPolicy
from thinkbooster.strategies.strategy_base import StrategyCancelled
from thinkbooster.strategies.strategy_belt import StrategyBELT

SHA = "a" * 64


def candidate(text, complete=False, thinking_complete=False, **data):
    return SimpleNamespace(
        text=text,
        raw_text=text,
        token_ids=list(text.encode()),
        is_trajectory_complete=complete,
        is_thinking_complete=thinking_complete,
        other_data=data,
        output=None,
    )


def policy(threshold=2):
    return BeltStoppingPolicy(SHA, "H", (1,), 0, (0,), (1,), threshold)


class Scorer:
    checkpoint_sha256 = SHA

    def __init__(self):
        self.histories = []

    def score_prefix(self, chat, trajectory):
        self.histories.append([s.raw_text for s in trajectory])
        return BeltSignal(
            dict(q=0.7, v=0.6, r=0.5, stability=0.8),
            (float(len(trajectory)),),
            sum(len(s.token_ids) for s in trajectory),
            SHA,
        )


class Generator:
    thinking_mode = False

    def __init__(self, scripts, answers=None):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.answers = answers or {}
        self.answer_calls = []
        self.step_calls = []
        self.stats = {}

    def reset_per_sample_stats(self):
        self.stats.clear()

    def record(self, sample_id, value):
        self.stats[sample_id] = self.stats.get(sample_id, 0) + len(value.token_ids)

    def generate_step_candidates_batch(
        self, requests, trajectories, sample_ids, **kwargs
    ):
        self.step_calls.append(list(sample_ids))
        result = []
        for sample_id in sample_ids:
            value = self.scripts[sample_id].pop(0)
            if value is not None:
                self.record(sample_id, value)
            result.append([value] if value is not None else [])
        return result

    def generate_answer_candidates_batch(
        self, requests, trajectories, sample_ids, **kwargs
    ):
        self.answer_calls.append((sample_ids, [list(t) for t in trajectories]))
        result = []
        for sample_id in sample_ids:
            value = self.answers.get(sample_id)
            if value is not None:
                self.record(sample_id, value)
            result.append([value] if value is not None else [])
        return result

    def get_sample_stats_for(self, sample_id):
        return {"output_tokens": self.stats.get(sample_id, 0)}


def requests(n):
    return [[{"role": "user", "content": f"question {i}"}] for i in range(n)]


def test_heterogeneous_batch_order_stop_natural_failure_and_token_accounting():
    first, second = candidate("reason one\n\n"), candidate("reason two\n\n")
    final = candidate("\\boxed{42}", complete=True)
    natural = candidate("\\boxed{7}", complete=True)
    generator = Generator({42: [first, second], 7: [natural], 11: [None]}, {42: final})
    scorer = Scorer()
    results = StrategyBELT(
        generator, scorer, policy(), max_steps=3
    ).generate_trajectories_batch(requests(3), [42, 7, 11])
    assert [r["completion_reason"] for r in results] == [
        "belt_stop",
        "natural_completion",
        "generation_failed",
    ]
    assert [r["completed"] for r in results] == [True, True, False]
    assert generator.step_calls == [[42, 7, 11], [42]]
    assert generator.answer_calls[0][0] == [42]
    assert scorer.histories == [[first.raw_text], [first.raw_text, second.raw_text]]
    assert results[0]["token_stats"]["output_tokens"] == sum(
        len(x.token_ids) for x in [first, second, final]
    )
    assert results[1]["trajectory"] == natural.raw_text
    assert [d["stop"] for d in results[0]["metadata"]["decisions"]] == [False, True]


def test_query_cadence_keeps_unqueried_step_history():
    generator = Generator(
        {0: [candidate("a"), candidate("b")]}, {0: candidate("answer", True)}
    )
    scorer = Scorer()
    result = StrategyBELT(
        generator, scorer, policy(), min_steps=2, query_stride=2
    ).generate_trajectory(requests(1)[0])
    assert scorer.histories == [["a", "b"]]
    assert result["validity_scores"] == [None, 0.7, None]
    assert result["completed"]


def test_max_steps_does_not_claim_a_completed_answer():
    generator = Generator({0: [candidate("a"), candidate("b")]})
    result = StrategyBELT(
        generator, Scorer(), policy(10), max_steps=2
    ).generate_trajectory(requests(1)[0])
    assert result["completion_reason"] == "max_steps"
    assert result["max_steps_hit"] and not result["completed"]
    assert not generator.answer_calls


def test_missing_answer_is_incomplete():
    generator = Generator({0: [candidate("reason")]})
    result = StrategyBELT(generator, Scorer(), policy(0)).generate_trajectory(
        requests(1)[0]
    )
    assert not result["completed"]
    assert result["completion_reason"] == "answer_generation_failed"


def test_thinking_close_still_generates_answer_with_hf_completion_flag():
    generator = Generator(
        {0: [candidate("done</think>", complete=True, thinking_complete=True)]},
        {0: candidate("\\boxed{4}", True)},
    )
    generator.thinking_mode = True
    result = StrategyBELT(generator, Scorer(), policy()).generate_trajectory(
        requests(1)[0]
    )
    assert len(generator.answer_calls) == 1
    assert result["trajectory"] == "done</think>\\boxed{4}"
    assert result["completed"]


def test_context_limit_is_not_natural_completion():
    generator = Generator(
        {0: [candidate("partial", complete=True, completion_reason="context_limit")]}
    )
    result = StrategyBELT(generator, Scorer(), policy()).generate_trajectory(
        requests(1)[0]
    )
    assert result["completion_reason"] == "context_limit"
    assert result["context_limit_hit"] and not result["completed"]


def test_length_limited_answer_is_incomplete_even_when_generator_marks_complete():
    answer = candidate("truncated answer", complete=True)
    answer.output = SimpleNamespace(finish_reason="length")
    generator = Generator({0: [candidate("reason")]}, {0: answer})
    result = StrategyBELT(generator, Scorer(), policy(0)).generate_trajectory(
        requests(1)[0]
    )
    assert not result["completed"]
    assert result["completion_reason"] == "answer_incomplete"


def test_policy_stop_closes_thinking_and_records_control_text():
    generator = Generator(
        {0: [candidate("reason")]}, {0: candidate("\\boxed{4}", True)}
    )
    generator.thinking_mode = True
    result = StrategyBELT(generator, Scorer(), policy(0)).generate_trajectory(
        requests(1)[0]
    )
    assert result["trajectory"] == "reason\n</think>\n\n\\boxed{4}"
    assert result["metadata"]["injected_control_text"] == ["\n</think>\n\n"]
    assert generator.answer_calls[0][1][0][-1].text == "\n</think>\n\n"
    assert result["validity_scores"] == [0.7, None, None]
    assert result["reasoning_steps"] == 1
    assert result["token_stats"]["output_tokens"] == len("reason\\boxed{4}".encode())


def test_cancelled_batch_does_not_generate():
    generator = Generator({0: [candidate("reason")]})
    strategy = StrategyBELT(generator, Scorer(), policy())
    event = threading.Event()
    event.set()
    strategy.set_cancel_event(event)
    with pytest.raises(StrategyCancelled):
        strategy.generate_trajectory(requests(1)[0])
    assert not generator.step_calls


def test_invalid_batch_and_parameters_fail():
    generator = Generator({})
    with pytest.raises(ValueError):
        StrategyBELT(generator, Scorer(), policy(), max_steps=0)
    strategy = StrategyBELT(generator, Scorer(), policy())
    with pytest.raises(ValueError):
        strategy.generate_trajectories_batch(requests(2), [1, 1])
    with pytest.raises(ValueError):
        strategy.generate_trajectories_batch(requests(2), [1])
    assert strategy.generate_trajectories_batch([]) == []


def test_unimplemented_batch_api_is_rejected():
    from thinkbooster.generators.base import StepCandidateGeneratorBase

    with pytest.raises(ValueError, match="batched step and answer"):
        StrategyBELT(StepCandidateGeneratorBase(1), Scorer(), policy())


def test_wrong_probe_fails_before_any_generation():
    generator = Generator({0: [candidate("reason")]})
    scorer = Scorer()
    scorer.checkpoint_sha256 = "b" * 64
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        StrategyBELT(generator, scorer, policy())
    assert generator.step_calls == []
    assert scorer.histories == []
