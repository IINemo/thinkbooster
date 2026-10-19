import pytest

vllm = pytest.importorskip("vllm")

from thinkbooster.generators.vllm import VLLMStepGenerator  # noqa: E402


def bare_generator():
    generator = object.__new__(VLLMStepGenerator)
    generator.generation_limit = 64
    generator.temperature = 0.6
    generator.top_p = 0.95
    generator.top_k = 20
    generator.stop_token_ids = None
    generator.presence_penalty = 0.0
    return generator


def test_stop_strings_are_stripped_unless_requested():
    generator = bare_generator()
    default = generator._create_sampling_params(stop_tokens=["\n\n"])
    kept = generator._create_sampling_params(
        stop_tokens=["\n\n"], include_stop_str_in_output=True
    )
    assert default.include_stop_str_in_output is False
    assert kept.include_stop_str_in_output is True
    assert kept.stop == ["\n\n"]


def test_batch_api_forwards_the_option(monkeypatch):
    generator = bare_generator()
    seen = {}
    monkeypatch.setattr(
        generator,
        "_generate_step_candidates_impl",
        lambda *args, **kwargs: seen.update(kwargs) or [],
    )
    generator.generate_step_candidates_batch(
        [[]], [[]], include_stop_str_in_output=True
    )
    assert seen["include_stop_str_in_output"] is True


def length_cut_generator(text):
    from types import SimpleNamespace

    from thinkbooster.generators.base import StepCandidateGeneratorBase

    generator = bare_generator()
    StepCandidateGeneratorBase.__init__(generator, generation_batch_size=1)
    tokenizer = SimpleNamespace(
        encode=lambda s, **kw: list(s.encode()),
        decode=lambda ids, **kw: bytes(ids).decode(),
        apply_chat_template=lambda chat, **kw: "<prompt>",
    )
    output = SimpleNamespace(
        text=text,
        token_ids=list(text.encode()),
        logprobs=[{}] * len(text),
        stop_reason=None,
        finish_reason="length",
    )
    generator.model = SimpleNamespace(
        generate=lambda prompts, params: [SimpleNamespace(outputs=[output])]
    )
    generator.tokenizer = tokenizer
    generator.thinking_mode = True
    generator.stop_tokens = ["\n\n", "</think>"]
    generator.step_token_limit = len(text)  # the step ends on the cap
    generator.min_step_tokens = 0
    generator.seed = None
    generator.reasoning_effort = None
    generator.context_budget = 10**6
    generator.flop_calculator = None
    return generator


@pytest.mark.parametrize("trim, expected", [(True, "One. "), (False, "One. Two thr")])
def test_length_cut_step_is_kept_whole_on_request(trim, expected):
    generator = length_cut_generator("One. Two thr")
    [[step]] = generator.generate_step_candidates_batch(
        [[{"role": "user", "content": "q"}]],
        [[]],
        compute_uncertainty=False,
        truncate_at_sentence=trim,
    )
    assert step.raw_text.strip() == expected.strip()
