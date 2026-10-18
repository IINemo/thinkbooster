"""Opt-in CPU parity check using the published reader and deterministic features."""

import copy
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from thinkbooster.belt.runtime import BeltRuntime, download_checkpoint


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("BELT_TEST_CACHE"),
    reason="Set BELT_TEST_CACHE to enable checkpoint downloads",
)
def test_published_reader_matches_original_forward_with_all_end_markers():
    from rl_controller.rollout_contract import resolve_native_root
    from rl_controller.scripts import rl_train
    from transformers import Qwen3Config

    sys.path.insert(0, str(resolve_native_root()))
    from luh import AutoUncertaintyHead

    weight, template, record = download_checkpoint(
        cache_dir=os.environ["BELT_TEST_CACHE"]
    )
    payload = torch.load(weight, map_location="cpu", weights_only=True)
    base = SimpleNamespace(config=Qwen3Config(hidden_size=2048, num_hidden_layers=28))
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


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("BELT_TEST_CACHE"),
    reason="Set BELT_TEST_CACHE to enable checkpoint downloads",
)
def test_qwen_tokenizer_scores_steps_cut_at_stripped_stop_strings():
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
    # Real cut points: vLLM strips sentence-start stop strings such as "\nSo ".
    steps = ["Okay, so x = 5.\n", "\nSo the area is 25.\n", "\nLet me verify this."]
    probe.score_prefix(chat, steps)
    completion_ids, ends = calls[-1]
    assert completion_ids == tokenizer.encode("".join(steps), add_special_tokens=False)
    assert len(ends) == 3 and ends == sorted(set(ends))
    assert ends[-1] == len(completion_ids)

    # Where retokenizing each prefix is stable, both methods place the same ends.
    stable = ["Okay, so x = 5.\n\n", "Wait, let me check.", " Yes."]
    probe.score_prefix(chat, stable)
    expected = [
        len(tokenizer.encode("".join(stable[: i + 1]), add_special_tokens=False))
        for i in range(len(stable))
    ]
    assert calls[-1][1] == expected
