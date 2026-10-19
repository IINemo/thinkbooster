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
    thinking_mode = True

    def __init__(self, scripts, answers=None):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.answers = answers or {}
        self.answer_calls = []
        self.step_calls = []
        self.step_kwargs = []
        self.stats = {}

    def reset_per_sample_stats(self):
        self.stats.clear()

    def record(self, sample_id, value):
        self.stats[sample_id] = self.stats.get(sample_id, 0) + len(value.token_ids)

    def generate_step_candidates_batch(
        self, requests, trajectories, sample_ids, **kwargs
    ):
        self.step_calls.append(list(sample_ids))
        self.step_kwargs.append(kwargs)
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
        generator, scorer, policy(), max_steps=3, min_steps=1
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


def test_query_cadence_counts_blocks_and_keeps_unqueried_history():
    blocks = [candidate(f"{x}\n\n") for x in "abcd"]
    close = candidate("e</think>", thinking_complete=True)
    generator = Generator({0: blocks + [close]}, {0: candidate("answer", True)})
    scorer = Scorer()
    result = StrategyBELT(
        generator, scorer, policy(10), query_stride=2
    ).generate_trajectory(requests(1)[0])
    texts = [b.raw_text for b in blocks]
    # Block 1 is never a decision; stride 2 then queries blocks 2 and 4.
    assert scorer.histories == [texts[:2], texts]
    assert [d["block"] for d in result["metadata"]["decisions"]] == [2, 4]
    # Blocks a-d, the closing step, then the answer.
    assert result["validity_scores"] == [None, 0.7, None, 0.7, None, None]
    assert result["completed"]


def test_steps_request_blocks_with_the_blank_line_kept():
    generator = Generator({0: [candidate("a\n\n"), candidate("b")]})
    StrategyBELT(generator, Scorer(), policy(10), max_steps=2).generate_trajectory(
        requests(1)[0]
    )
    assert generator.step_kwargs[0]["stop_tokens_override"] == ["\n\n", "</think>"]
    assert generator.step_kwargs[0]["include_stop_str_in_output"] is True
    assert generator.step_kwargs[0]["truncate_at_sentence"] is False


def test_only_new_block_ends_are_decisions():
    steps = ["first\n\n", "second half", " done\n\n", "\n\n", "third \n\t\n"]
    generator = Generator({0: [candidate(x) for x in steps]})
    result = StrategyBELT(
        generator, Scorer(), policy(10), min_steps=1, max_steps=5
    ).generate_trajectory(requests(1)[0])
    decisions = result["metadata"]["decisions"]
    # Not after the mid-block cut, nor after the whitespace-only step.
    assert [(d["block"], d["step"]) for d in decisions] == [(1, 1), (2, 3), (3, 5)]
    assert result["completion_reason"] == "max_steps"


def test_blocks_past_max_blocks_are_not_queried():
    generator = Generator({0: [candidate(f"{x}\n\n") for x in "abc"]})
    result = StrategyBELT(
        generator, Scorer(), policy(10), min_steps=1, max_blocks=2, max_steps=3
    ).generate_trajectory(requests(1)[0])
    assert [d["block"] for d in result["metadata"]["decisions"]] == [1, 2]


def test_max_steps_does_not_claim_a_completed_answer():
    generator = Generator({0: [candidate("a"), candidate("b")]})
    result = StrategyBELT(
        generator, Scorer(), policy(10), max_steps=2
    ).generate_trajectory(requests(1)[0])
    assert result["completion_reason"] == "max_steps"
    assert result["max_steps_hit"] and not result["completed"]
    assert not generator.answer_calls


def test_missing_answer_is_incomplete():
    generator = Generator({0: [candidate("reason\n\n")]})
    result = StrategyBELT(
        generator, Scorer(), policy(0), min_steps=1
    ).generate_trajectory(requests(1)[0])
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
    generator = Generator({0: [candidate("reason\n\n")]}, {0: answer})
    result = StrategyBELT(
        generator, Scorer(), policy(0), min_steps=1
    ).generate_trajectory(requests(1)[0])
    assert not result["completed"]
    assert result["completion_reason"] == "answer_incomplete"


def test_policy_stop_closes_thinking_and_records_control_text():
    generator = Generator(
        {0: [candidate("reason\n\n")]}, {0: candidate("\\boxed{4}", True)}
    )
    generator.thinking_mode = True
    result = StrategyBELT(
        generator, Scorer(), policy(0), min_steps=1
    ).generate_trajectory(requests(1)[0])
    assert result["trajectory"] == "reason\n\n\n</think>\n\n\\boxed{4}"
    assert result["metadata"]["injected_control_text"] == ["\n</think>\n\n"]
    assert generator.answer_calls[0][1][0][-1].text == "\n</think>\n\n"
    assert result["validity_scores"] == [0.7, None, None]
    assert result["reasoning_steps"] == 1
    assert result["token_stats"]["output_tokens"] == len(
        "reason\n\n\\boxed{4}".encode()
    )


def test_cancelled_batch_does_not_generate():
    generator = Generator({0: [candidate("reason")]})
    strategy = StrategyBELT(generator, Scorer(), policy())
    event = threading.Event()
    event.set()
    strategy.set_cancel_event(event)
    with pytest.raises(StrategyCancelled):
        strategy.generate_trajectory(requests(1)[0])
    assert not generator.step_calls


@pytest.mark.parametrize(
    "kwargs",
    [dict(max_steps=0), dict(min_steps=0), dict(min_steps=3, max_blocks=2)],
)
def test_invalid_step_and_block_limits_fail(kwargs):
    with pytest.raises(ValueError):
        StrategyBELT(Generator({}), Scorer(), policy(), **kwargs)


def test_minimum_step_length_is_rejected():
    generator = Generator({})
    generator.min_step_tokens = 50
    with pytest.raises(ValueError, match="min_step_tokens=0"):
        StrategyBELT(generator, Scorer(), policy())


def test_generator_without_per_call_stop_options_is_rejected():
    class Fixed(Generator):
        def generate_step_candidates_batch(self, requests, trajectories, sample_ids):
            return []

    with pytest.raises(ValueError, match="include_stop_str_in_output"):
        StrategyBELT(Fixed({}), Scorer(), policy())


def test_invalid_batch_and_parameters_fail():
    generator = Generator({})
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


def test_backend_metadata_and_counters_preserve_decisions_and_cost_scope():
    generator = Generator(
        {0: [candidate("reason\n\n")]}, {0: candidate("answer", True)}
    )
    resets = []
    generator.model = SimpleNamespace(reset_hs_step_cache=lambda: resets.append(True))
    scorer = Scorer()
    scorer.metadata = {
        "scoring_backend": "vllm",
        "backend_stats": {"repair_calls": 2, "queries": 1},
    }
    result = StrategyBELT(
        generator, scorer, policy(0), min_steps=1
    ).generate_trajectory(requests(1)[0])
    assert resets == [True]
    assert result["metadata"]["scoring_backend"] == "vllm"
    assert result["metadata"]["backend_stats"]["repair_calls"] == 2
    assert result["metadata"]["backend_stats_scope"] == "strategy_batch"
    assert result["metadata"]["decisions"][0]["stop"] is True
    assert result["metadata"]["scoring_cost_in_token_stats"] is False
    assert result["completed"] and result["completion_reason"] == "belt_stop"


@pytest.mark.parametrize(
    "attrs", [{"thinking_mode": False}, {"disable_thinking_mode": True}]
)
def test_non_thinking_generator_is_rejected(attrs):
    generator = Generator({})
    for name, value in attrs.items():
        setattr(generator, name, value)
    with pytest.raises(ValueError, match="thinking-mode generator"):
        StrategyBELT(generator, Scorer(), policy())
