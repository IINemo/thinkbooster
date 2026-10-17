from types import SimpleNamespace

import pytest
import torch

from thinkbooster.belt.runtime import BeltRuntime, download_checkpoint


class CharacterTokenizer:
    def apply_chat_template(self, chat, **kwargs):
        return [101, 102, 103]

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


class State:
    def __init__(self):
        self.calls = []
        self.cleared = 0

    def __call__(self, prompt_ids, completion_ids, block, block_ends):
        self.calls.append((prompt_ids, completion_ids, block, block_ends))
        return {"repr": torch.tensor([1.0, 2.0])}

    def clear_store(self):
        self.cleared += 1


def runtime():
    readout = torch.nn.Linear(2, 4)
    with torch.no_grad():
        readout.weight.copy_(torch.tensor([[0, 0], [1, 0], [0, -1], [1, 1]]))
        readout.bias.zero_()
    return BeltRuntime(CharacterTokenizer(), State(), readout, "a" * 64)


def test_readout_targets_and_full_boundary_history():
    probe = runtime()
    result = probe.score_prefix(
        [{"role": "user", "content": "question"}], ["ab", "cd", "ef"]
    )
    assert probe.state_fn.calls == [
        ([101, 102, 103], [97, 98, 99, 100, 101, 102], 3, [2, 4, 6])
    ]
    assert result.hidden == (1, 2)
    assert result.beliefs["q"] == 0.5
    assert result.beliefs["v"] == pytest.approx(torch.sigmoid(torch.tensor(1.0)).item())
    assert result.beliefs["r"] == pytest.approx(
        torch.sigmoid(torch.tensor(-2.0)).item()
    )
    assert result.beliefs["stability"] == pytest.approx(
        torch.sigmoid(torch.tensor(3.0)).item()
    )
    assert result.prefix_tokens == 6
    assert probe.state_fn.cleared == 1


@pytest.mark.parametrize("ends", [[2, 2, 6], [0, 6], [4, 2, 6], [2, 5], []])
def test_misaligned_block_ends_fail(ends):
    with pytest.raises(ValueError):
        runtime().score_token_ids([101], [1, 2, 3, 4, 5, 6], ends)


def test_reader_failure_clears_cached_state():
    probe = runtime()
    probe.readout = lambda hidden: (_ for _ in ()).throw(
        RuntimeError("readout failure")
    )
    with pytest.raises(RuntimeError, match="readout failure"):
        probe.score_token_ids([101], [1, 2], [2])
    assert probe.state_fn.cleared == 1


def test_text_steps_must_preserve_tokenization():
    probe = runtime()
    probe.tokenizer.encode = lambda text, **kw: [1] if len(text) == 1 else [2, 3]
    with pytest.raises(ValueError, match="changes tokenization"):
        probe.score_prefix([{"role": "user", "content": "q"}], ["a", "b"])


def test_step_objects_are_supported():
    probe = runtime()
    probe.score_prefix(
        [{"role": "user", "content": "q"}], [SimpleNamespace(raw_text="ab", text="ab")]
    )
    assert probe.state_fn.calls[0][-1] == [2]


@pytest.mark.parametrize(
    "kwargs", [dict(seed=-1), dict(seed=True), dict(checkpoint_repo="unpublished")]
)
def test_unknown_checkpoint_selection_fails_before_network(kwargs):
    if "checkpoint_repo" in kwargs:
        kwargs["repo_id"] = kwargs.pop("checkpoint_repo")
    with pytest.raises(ValueError):
        download_checkpoint(**kwargs)
