"""Opt-in CPU parity check using the published reader and deterministic features."""

import copy
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from thinkbooster.belt.runtime import (
    BeltRuntime,
    _native_reader_imports,
    download_checkpoint,
)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("BELT_TEST_CACHE"),
    reason="Set BELT_TEST_CACHE to enable checkpoint downloads",
)
def test_published_reader_matches_original_forward_with_all_end_markers():
    from rl_controller.rollout_contract import resolve_native_root
    from rl_controller.scripts import rl_train
    from transformers import Qwen3Config

    weight, template, record = download_checkpoint(
        cache_dir=os.environ["BELT_TEST_CACHE"]
    )
    payload = torch.load(weight, map_location="cpu", weights_only=True)
    base = SimpleNamespace(config=Qwen3Config(hidden_size=2048, num_hidden_layers=28))
    before_path = sys.path[:]
    before_modules = {
        name: module
        for name, module in sys.modules.copy().items()
        if name.split(".")[0] in ("luh", "train_luh")
    }
    with _native_reader_imports(resolve_native_root()):
        from luh import AutoUncertaintyHead

        head = AutoUncertaintyHead.from_pretrained(str(template), base).eval()
    head.load_state_dict(payload["head"], strict=True)
    frozen = copy.deepcopy(head).eval()
    prompt_ids, completion_ids, ends = [11, 12, 13], [21, 22, 23, 24, 25, 26], [2, 4, 6]
    length = len(prompt_ids) + len(completion_ids) - 1
    features = torch.sin(
        torch.arange(length * 2048, dtype=torch.float32).reshape(1, length, 2048) / 1000
    )

    def get_features(sequences, device):
        count = len(sequences[0]) - 1
        return (
            features[:, :count].to(device),
            torch.ones(1, count, dtype=torch.long),
            [0],
            [],
        )

    provider = SimpleNamespace(get_features=get_features, chunk_size=0)
    with patch.object(
        rl_train,
        "_load_head_runtime",
        return_value=(object(), provider, head, frozen, 2048),
    ):
        runtime = BeltRuntime.from_pretrained(
            device="cpu", dtype="float32", cache_dir=os.environ["BELT_TEST_CACHE"]
        )
    actual = runtime.score_token_ids(prompt_ids, completion_ids, ends)

    captured = []
    hook = head.classifier.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].clone())
    )
    # Independent construction: feature column = prompt length + exclusive end - 2.
    masks = torch.zeros(3, length, dtype=torch.long)
    for row, end in enumerate(ends):
        column = len(prompt_ids) + end - 2
        masks[row, column - 1 : column + 1] = 1
    with torch.inference_mode():
        head.forward_from_features(
            features, torch.ones(1, length, dtype=torch.long), [masks]
        )
        expected_hidden = captured[-1][-1]
        expected_probabilities = torch.sigmoid(runtime.readout(expected_hidden))
    hook.remove()
    assert actual.checkpoint_sha256 == record["sha256"]
    torch.testing.assert_close(
        torch.tensor(actual.hidden), expected_hidden, rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        torch.tensor(list(actual.beliefs.values())),
        expected_probabilities,
        rtol=0,
        atol=1e-6,
    )
    assert sys.path == before_path
    assert {
        name: module
        for name, module in sys.modules.copy().items()
        if name.split(".")[0] in ("luh", "train_luh")
    } == before_modules


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("BELT_TEST_CACHE"),
    reason="Set BELT_TEST_CACHE to enable checkpoint downloads",
)
def test_qwen_tokenizer_places_an_end_after_every_blank_line_block():
    from transformers import AutoTokenizer

    from thinkbooster.belt.runtime import BACKBONE, BACKBONE_REVISION

    tokenizer = AutoTokenizer.from_pretrained(
        BACKBONE, revision=BACKBONE_REVISION, cache_dir=os.environ["BELT_TEST_CACHE"]
    )
    calls = []

    class State:
        def __call__(self, prompt_ids, completion_ids, blocks, block_ends):
            calls.append((completion_ids, block_ends))
            return {"repr": torch.zeros(4)}

        def clear_store(self):
            pass

    probe = BeltRuntime(tokenizer, State(), torch.nn.Linear(4, 4), "a" * 64)
    chat = [{"role": "user", "content": "What is x?"}]
    blocks = [
        "Okay, so x = 5.\n\n",
        "So the area is 25.\n\n",
        "Let me verify this.\n\n",
    ]
    text = "".join(blocks)
    expected = [
        len(tokenizer.encode("".join(blocks[: i + 1]), add_special_tokens=False))
        for i in range(len(blocks))
    ]
    # One step per block, as the strategy generates them.
    probe.score_prefix(chat, blocks)
    assert calls[-1] == (tokenizer.encode(text, add_special_tokens=False), expected)
    # Any other cut of the same text places the same ends.
    cuts = [8, 23, 36]  # mid-block, mid-block, between the "\n" of a blank line
    pieces = [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]
    probe.score_prefix(chat, pieces)
    assert calls[-1][1] == expected


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("BELT_TEST_CACHE"), reason="Set BELT_TEST_CACHE")
def test_injected_provider_loads_only_reader_and_matches_original_head(monkeypatch):
    from rl_controller.scripts import rl_train
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, Qwen3Config

    def forbidden(*args, **kwargs):
        raise AssertionError("HF backbone/runtime must not load for injected features")

    monkeypatch.setattr(rl_train, "_load_head_runtime", forbidden)
    monkeypatch.setattr(AutoModel, "from_pretrained", forbidden)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(AutoConfig, "from_pretrained", forbidden)
    features = torch.sin(torch.arange(8 * 2048).reshape(1, 8, 2048) / 1000)
    provider = SimpleNamespace(
        get_features=lambda sequences, device: (
            features[:, : len(sequences[0]) - 1].to(device),
            torch.ones(1, len(sequences[0]) - 1, dtype=torch.long, device=device),
            [0],
            [],
        ),
        chunk_size=0,
        backend="vllm",
        stats={"repair_calls": 0},
    )
    tokenizer = object()
    runtime = BeltRuntime.from_pretrained(
        device="cpu",
        dtype="float32",
        cache_dir=os.environ["BELT_TEST_CACHE"],
        feature_provider=provider,
        tokenizer=tokenizer,
        base_config=Qwen3Config(hidden_size=2048, num_hidden_layers=28),
    )
    assert runtime.tokenizer is tokenizer and runtime.state_fn.provider is provider
    signal = runtime.score_token_ids([11, 12, 13], [21, 22, 23, 24, 25, 26], [2, 4, 6])
    actual_hidden = torch.tensor(signal.hidden)
    captured = []
    hook = runtime.state_fn.head.classifier.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].clone())
    )
    masks = torch.zeros(3, 8, dtype=torch.long)
    for row, end in enumerate([2, 4, 6]):
        column = 3 + end - 2
        masks[row, column - 1 : column + 1] = 1
    with torch.inference_mode():
        runtime.state_fn.head.forward_from_features(features, torch.ones(1, 8), [masks])
    hook.remove()
    torch.testing.assert_close(actual_hidden, captured[-1][-1], rtol=0, atol=1e-6)
    assert runtime.metadata["scoring_backend"] == "vllm"
    assert runtime.metadata["backend_stats"] == {"repair_calls": 0}


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("BELT_TEST_CACHE"), reason="Set BELT_TEST_CACHE")
def test_example_prompt_matches_belt_chain_prompt_token_for_token():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from rl_controller.rollout_contract import prompt_ids_for
    from transformers import AutoTokenizer

    from thinkbooster.belt.runtime import BACKBONE, BACKBONE_REVISION
    from thinkbooster.strategies.strategy_belt import _belt_chat

    tokenizer = AutoTokenizer.from_pretrained(
        BACKBONE, revision=BACKBONE_REVISION, cache_dir=os.environ["BELT_TEST_CACHE"]
    )
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base=None, config_dir=str(root / "config")):
        cfg = compose(config_name="experiments/belt/math500_qwen3_1p7b")
    from belt_requests import built_request

    question = "What is the smallest positive integer $n$ with $n^2 > 50$?"
    # The request as run_tts_eval builds it, from its own prompt lookup.
    request = built_request(cfg, question)
    ours = tokenizer.apply_chat_template(
        _belt_chat(request),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    belt = prompt_ids_for(
        tokenizer,
        question,
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
        decode=lambda ids: tokenizer.decode(ids),
    )
    assert list(ours) == list(belt)
