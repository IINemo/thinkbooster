from types import SimpleNamespace

import numpy as np
import pytest
import torch

from thinkbooster.belt.vllm_backend import BeltVLLMBackend
from thinkbooster.belt.vllm_wire import encode_features


def output(key, prompt, completion):
    return SimpleNamespace(
        request_id=key,
        prompt_token_ids=prompt,
        outputs=[SimpleNamespace(token_ids=completion)],
    )


def capture(positions, values=None):
    if values is None:
        values = positions
    return {
        "positions": positions,
        "features": encode_features(
            np.repeat(np.asarray(values, dtype=np.float32)[:, None], 2048, axis=1)
        ),
    }


class Engine:
    def __init__(self, calls):
        self.calls = list(calls)
        self.rpc_names = []
        self.resets = 0
        self.generated = []
        self.llm_engine = SimpleNamespace(
            engine_core=SimpleNamespace(collective_rpc=self.rpc)
        )

    def rpc(self, name):
        self.rpc_names.append(name)
        return [self.captured if name == "belt_get_capture" else None]

    def generate(self, *args, **kwargs):
        self.generated.append((args, kwargs))
        outputs, self.captured = self.calls.pop(0)
        return outputs

    def reset_prefix_cache(self):
        self.resets += 1
        return True

    def get_tokenizer(self):
        return "tokenizer"


def backend(calls, monkeypatch, **kwargs):
    import thinkbooster.belt.vllm_backend as module

    monkeypatch.setattr(
        module,
        "_repair_sampling_params",
        lambda: SimpleNamespace(n=1, max_tokens=1, temperature=0, seed=0, stop=None),
    )
    engine = Engine(calls)
    return BeltVLLMBackend(engine, **kwargs), engine


def test_generation_is_transparent_and_does_not_expose_llm_bypass(monkeypatch):
    item = output("r", [10, 11], [12, 13])
    model, engine = backend([([item], {"r": capture([0, 1, 2])})], monkeypatch)
    assert getattr(model, "llm", model) is model
    assert model.generate(["text"], SimpleNamespace(n=1)) == [item]
    features, attention, keep, skipped = model.get_features([[10, 11, 12, 13]], "cpu")
    assert features.shape == (1, 3, 2048)
    torch.testing.assert_close(features[0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert attention.tolist() == [[1, 1, 1]] and keep == [0] and skipped == []
    assert model.stats["cache_hits"] == 1 and model.stats["repair_calls"] == 0
    assert model.metadata["feature_cache_ownership"] == "first_cached_causal_prefix"
    assert engine.rpc_names == [
        "belt_setup_capture",
        "belt_reset_capture",
        "belt_get_capture",
    ]


def test_changed_tail_is_invalidated_and_interleaved_requests_do_not_mix(monkeypatch):
    model, engine = backend(
        [
            (
                [output("b", [50], [51, 52]), output("a", [10], [11, 12])],
                {"a": capture([0, 1], [100, 101]), "b": capture([0, 1], [200, 201])},
            ),
            (
                [output("repair", [10, 99, 13], [14])],
                {"repair": capture([1, 2], [999, 1000])},
            ),
        ],
        monkeypatch,
    )
    model.generate(["a", "b"])
    b = model.get_features([[50, 51, 52]], "cpu")[0]
    assert b[0, :, 0].tolist() == [200, 201]
    result = model.get_features([[10, 99, 13]], "cpu")[0]
    assert result[0, :, 0].tolist() == [100, 999]
    assert model.stats["repair_calls"] == 1 and engine.resets == 0
    assert engine.generated[1][0][0] == [{"prompt_token_ids": [10, 99, 13]}]


def test_apc_donor_prefix_merging_retains_only_causal_common_prefix(monkeypatch):
    model, _ = backend(
        [
            ([output("a", [1, 2], [3, 4])], {"a": capture([0, 1, 2], [10, 20, 30])}),
            ([output("b", [1, 2, 9], [8, 7])], {"b": capture([2, 3], [90, 80])}),
        ],
        monkeypatch,
    )
    model.generate(["a"])
    model.generate(["b"])
    result = model.get_features([[1, 2, 9, 8, 7]], "cpu")[0]
    assert result[0, :, 0].tolist() == [10, 20, 90, 80]
    assert model.stats["repair_calls"] == 0


@pytest.mark.parametrize("same_call", [False, True])
def test_same_tokens_retain_first_cached_rows_and_count_all_captures(
    monkeypatch, same_call
):
    first = output("first", [10, 11], [12, 13])
    later = output("later", [10, 11], [12, 13])
    captures = {
        "first": capture([0, 1, 2], [10, 20, 30]),
        "later": capture([0, 1, 2], [100, 200, 300]),
    }
    calls = (
        [([first, later], captures)]
        if same_call
        else [
            ([item], {item.request_id: captures[item.request_id]})
            for item in [first, later]
        ]
    )
    model, _ = backend(calls, monkeypatch)
    for _ in calls:
        model.generate(["prompt"])
    result = model.get_features([[10, 11, 12, 13]], "cpu")[0]
    assert result[0, :, 0].tolist() == [10, 20, 30]
    assert model.stats["generation_captured_tokens"] == 6
    assert model.stats["repair_calls"] == 0


@pytest.mark.parametrize("max_cache_entries", [1, 16])
def test_longer_prefix_inherits_owned_rows_before_bounded_cache_eviction(
    monkeypatch, max_cache_entries
):
    model, _ = backend(
        [
            (
                [output("short", [10, 11], [12, 13])],
                {"short": capture([0, 1, 2], [10, 20, 30])},
            ),
            (
                [output("long", [10, 11, 12, 13], [14, 15])],
                {"long": capture([0, 1, 2, 3, 4], [100, 200, 300, 40, 50])},
            ),
        ],
        monkeypatch,
        max_cache_entries=max_cache_entries,
    )
    model.generate(["short"])
    cold = model.get_features([[10, 11, 12, 13]], "cpu")[0]
    model.generate(["long"])
    warm = model.get_features([[10, 11, 12, 13]], "cpu")[0]
    torch.testing.assert_close(warm, cold, rtol=0, atol=0)
    result = model.get_features([[10, 11, 12, 13, 14, 15]], "cpu")[0]
    assert result[0, :, 0].tolist() == [10, 20, 30, 40, 50]
    if max_cache_entries == 1:
        assert (10, 11, 12, 13) not in model._cache
    assert model.stats["generation_captured_tokens"] == 8
    assert model.stats["repair_calls"] == 0


def test_overlapping_capture_fills_sparse_holes_without_replacing_owned_rows(
    monkeypatch,
):
    model, _ = backend(
        [
            (
                [output("sparse", [1, 2], [3, 4, 5])],
                {"sparse": capture([0, 2], [10, 30])},
            ),
            (
                [output("fill", [1, 2, 3, 4], [5, 6])],
                {"fill": capture([0, 1, 2, 3, 4], [100, 20, 300, 40, 50])},
            ),
        ],
        monkeypatch,
    )
    model.generate(["sparse"])
    assert model._cache[(1, 2, 3, 4, 5)].mask.tolist() == [
        True,
        False,
        True,
        False,
        False,
    ]
    model.generate(["fill"])
    shorter = model.get_features([[1, 2, 3, 4, 5]], "cpu")[0]
    longer = model.get_features([[1, 2, 3, 4, 5, 6]], "cpu")[0]
    assert shorter[0, :, 0].tolist() == [10, 20, 30, 40]
    assert longer[0, :, 0].tolist() == [10, 20, 30, 40, 50]
    assert model.stats["generation_captured_tokens"] == 7
    assert model.stats["repair_calls"] == 0


def test_changed_token_tail_gets_new_rows_while_common_prefix_keeps_ownership(
    monkeypatch,
):
    model, _ = backend(
        [
            (
                [output("original", [1, 2], [3, 4])],
                {"original": capture([0, 1, 2], [10, 20, 30])},
            ),
            (
                [output("changed", [1, 2], [99, 8])],
                {"changed": capture([0, 1, 2], [100, 200, 990])},
            ),
        ],
        monkeypatch,
    )
    model.generate(["original"])
    model.generate(["changed"])
    original = model.get_features([[1, 2, 3, 4]], "cpu")[0]
    changed = model.get_features([[1, 2, 99, 8]], "cpu")[0]
    assert original[0, :, 0].tolist() == [10, 20, 30]
    assert changed[0, :, 0].tolist() == [10, 20, 990]
    assert model.stats["generation_captured_tokens"] == 6
    assert model.stats["repair_calls"] == 0


def test_eviction_of_all_causal_owners_allows_new_measurement(monkeypatch):
    model, _ = backend(
        [
            (
                [output("first", [10, 11], [12, 13])],
                {"first": capture([0, 1, 2], [10, 20, 30])},
            ),
            (
                [output("unrelated", [90, 91], [92])],
                {"unrelated": capture([0, 1], [900, 910])},
            ),
            (
                [output("recaptured", [10, 11], [12, 13])],
                {"recaptured": capture([0, 1, 2], [100, 200, 300])},
            ),
        ],
        monkeypatch,
        max_cache_entries=1,
    )
    model.generate(["first"])
    first = model.get_features([[10, 11, 12, 13]], "cpu")[0]
    model.generate(["unrelated"])
    assert list(model._cache) == [(90, 91, 92)]
    model.generate(["recaptured"])
    recaptured = model.get_features([[10, 11, 12, 13]], "cpu")[0]
    assert first[0, :, 0].tolist() == [10, 20, 30]
    assert recaptured[0, :, 0].tolist() == [100, 200, 300]
    assert model.stats["generation_captured_tokens"] == 8
    assert model.stats["repair_calls"] == 0


def test_repeated_absolute_position_keeps_first_row_but_counts_all_captures(
    monkeypatch,
):
    model, _ = backend(
        [
            (
                [output("r", [1, 2], [3, 4])],
                {"r": capture([2, 0, 1, 1], [30, 10, 20, 200])},
            )
        ],
        monkeypatch,
    )
    model.generate(["prompt"])
    result = model.get_features([[1, 2, 3, 4]], "cpu")[0]
    assert result[0, :, 0].tolist() == [10, 20, 30]
    assert model.stats["generation_captured_tokens"] == 4
    assert model.stats["repair_calls"] == 0


def test_unknown_apc_gaps_trigger_one_cold_retry_then_explicit_failure(monkeypatch):
    repair = [output("r", [1, 2, 3], [4])]
    model, engine = backend(
        [(repair, {"r": capture([2])}), (repair, {"r": capture([2])})], monkeypatch
    )
    with pytest.raises(RuntimeError, match="missing.*positions"):
        model.get_features([[1, 2, 3]], "cpu")
    assert engine.resets == 1 and len(engine.generated) == 2
    assert model.stats["repair_calls"] == 2
    assert model.stats["repair_tokens"] == 2
    assert model.stats["repair_requested_tokens"] == 6


def test_cold_retry_can_fill_gaps_and_cache_is_bounded_and_reset(monkeypatch):
    model, engine = backend(
        [
            ([output("r", [1, 2, 3], [4])], {"r": capture([2])}),
            ([output("s", [1, 2, 3], [4])], {"s": capture([0, 1, 2])}),
        ],
        monkeypatch,
        max_cache_entries=1,
    )
    assert model.get_features([[1, 2, 3]], "cpu")[0].shape == (1, 2, 2048)
    assert len(model._cache) == 1 and engine.resets == 1
    assert model.stats["repair_tokens"] == 4
    model.reset_hs_step_cache()
    assert len(model._cache) == 0 and model.stats["queries"] == 0


@pytest.mark.parametrize(
    "records,states",
    [
        ([output("a", [1], [2])], {"other": capture([0])}),
        ([output("a", [1], [2]), output("a", [3], [4])], {"a": capture([0])}),
        ([output("a", [1], [2])], {"a": capture([2])}),
    ],
)
def test_ambiguous_or_invalid_attribution_fails(records, states, monkeypatch):
    model, _ = backend([(records, states)], monkeypatch)
    with pytest.raises(RuntimeError):
        model.generate(["a"])


def test_n_greater_than_one_fails_before_generation(monkeypatch):
    model, engine = backend([], monkeypatch)
    with pytest.raises(ValueError, match="n=1"):
        model.generate(["a"], SimpleNamespace(n=2))
    assert engine.generated == []


def test_generator_compute_uncertainty_false_calls_capture_wrapper(monkeypatch):
    from thinkbooster.generators.vllm import VLLMStepGenerator

    item = output("r", [10, 11], [12, 13])
    item.outputs[0].text = "reason"
    item.outputs[0].stop_reason = "step"
    model, engine = backend([([item], {"r": capture([0, 1, 2])})], monkeypatch)
    generator = object.__new__(VLLMStepGenerator)
    generator.model = model
    generator.thinking_mode = True
    generator.tokenizer = SimpleNamespace(encode=lambda text: [10, 11])
    generator._build_prompt = lambda request, trajectory: "prompt"
    generator._create_sampling_params = lambda **kwargs: SimpleNamespace(n=1)
    generator._detect_line_repetitions = lambda text: False
    generator._process_generation_output = lambda **kwargs: SimpleNamespace(
        is_trajectory_complete=False, other_data={}, token_ids=[12, 13]
    )
    generator._record_generation = lambda *args, **kwargs: None
    generator.stop_tokens = ["step"]
    generator.step_token_limit = 100
    generator.min_step_tokens = 1
    generator.seed = 0
    generator.context_budget = 32768
    result = generator._generate_step_candidates_impl(
        [[]], [[]], candidates_per_step=1, compute_uncertainty=False
    )
    assert len(result[0]) == 1
    assert model.stats["generation_calls"] == 1 and len(engine.generated) == 1
    assert model.get_features([[10, 11, 12, 13]], "cpu")[0].shape == (1, 3, 2048)


def test_batch_reset_resets_apc_and_rejects_failed_reset(monkeypatch):
    model, engine = backend([], monkeypatch)
    model.reset_hs_step_cache()
    assert engine.resets == 1 and model.stats["prefix_cache_resets"] == 1
    model.reset_hs_step_cache(reset_prefix_cache=False)
    assert engine.resets == 1 and model.stats["prefix_cache_resets"] == 0
    engine.reset_prefix_cache = lambda: False
    with pytest.raises(RuntimeError, match="reset failed"):
        model.reset_hs_step_cache()
    engine.reset_prefix_cache = lambda: None
    with pytest.raises(RuntimeError, match="reset failed"):
        model.reset_hs_step_cache()


@pytest.mark.parametrize("lora_request", [object(), [object()], [None]])
def test_runtime_lora_requests_fail_before_generation_or_capture_reset(
    monkeypatch, lora_request
):
    model, engine = backend([], monkeypatch)
    initial_rpc_names = list(engine.rpc_names)
    with pytest.raises(ValueError, match="LoRA"):
        model.generate(["a"], SimpleNamespace(n=1), lora_request=lora_request)
    assert engine.generated == []
    assert engine.rpc_names == initial_rpc_names


def test_explicit_no_lora_request_preserves_base_generation(monkeypatch):
    item = output("r", [10, 11], [12])
    model, engine = backend([([item], {"r": capture([0, 1])})], monkeypatch)
    assert model.generate(["a"], lora_request=None) == [item]
    assert engine.generated[0][1]["lora_request"] is None


@pytest.mark.parametrize("as_batch", [False, True])
@pytest.mark.parametrize("keyword", [False, True])
def test_prompt_embeddings_fail_before_generation_or_capture_reset(
    monkeypatch, as_batch, keyword
):
    model, engine = backend([], monkeypatch)
    initial_rpc_names = list(engine.rpc_names)
    prompt = {"prompt_embeds": torch.ones(2, 2048)}
    prompts = [prompt] if as_batch else prompt
    args, kwargs = ((), {"prompts": prompts}) if keyword else ((prompts,), {})
    with pytest.raises(ValueError, match="prompt embeddings"):
        model.generate(*args, **kwargs)
    assert engine.generated == []
    assert engine.rpc_names == initial_rpc_names


@pytest.mark.parametrize("positions", [[True], [-1], [0.0], "0"])
def test_rpc_rejects_noncanonical_positions_before_cache_use(monkeypatch, positions):
    data = capture([0])
    data["positions"] = positions
    model, _ = backend([([output("r", [10], [11])], {"r": data})], monkeypatch)
    with pytest.raises(RuntimeError, match="positions"):
        model.generate(["a"])
    assert not model._cache


def test_rpc_rejects_row_count_mismatch_and_legacy_numpy_payload(monkeypatch):
    data = capture([0])
    data["positions"] = [0, 1]
    model, _ = backend([([output("r", [10], [11, 12])], {"r": data})], monkeypatch)
    with pytest.raises(RuntimeError, match="position/row count"):
        model.generate(["a"])
    data = {"positions": [0], "features": np.ones((1, 2048), dtype=np.float32)}
    model, _ = backend([([output("r", [10], [11])], {"r": data})], monkeypatch)
    with pytest.raises(RuntimeError, match="versioned byte envelope"):
        model.generate(["a"])


def test_rpc_applies_capture_bound_across_all_requests(monkeypatch):
    import thinkbooster.belt.vllm_backend as module

    monkeypatch.setattr(module, "MAX_CAPTURE_TOKENS", 2)
    records = [output("a", [1, 2], [3]), output("b", [4], [5])]
    model, _ = backend(
        [(records, {"a": capture([0, 1]), "b": capture([0])})], monkeypatch
    )
    with pytest.raises(RuntimeError, match="BELT feature wire"):
        model.generate(["a", "b"])
    assert not model._cache
