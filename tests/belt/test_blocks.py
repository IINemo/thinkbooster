import os
import random

import pytest

from thinkbooster.belt.blocks import block_count, block_end_offsets, ends_on_block


@pytest.mark.parametrize(
    "text, count, boundary",
    [
        ("", 0, False),
        ("\n\n", 0, False),
        ("a\n\n", 1, True),
        ("a\n\n\n", 1, True),
        ("a\n\nb", 2, False),
        ("a\n \t\nb\n\n", 2, True),
        ("a\n\n\n\nb\n\n", 2, True),
        ("a\n\n\n\n", 1, False),  # a stop that adds only a blank line
        ("a\n\ndone</think>\n\n", 2, False),
    ],
)
def test_block_count_and_boundary(text, count, boundary):
    assert block_count(text) == count
    assert ends_on_block(text) is boundary


@pytest.mark.skipif(
    not os.getenv("BELT_TEST_CACHE"), reason="Set BELT_TEST_CACHE (BELT on PYTHONPATH)"
)
def test_parity_with_belt_rollout_contract_on_random_texts():
    from rl_controller.rollout_contract import block_count as belt_count
    from rl_controller.rollout_contract import classify_stop
    from rl_controller.rollout_runner import block_ending_separators

    rng = random.Random(0)
    alphabet = ["a", "b", " ", "\t", "\n", "\n\n", "</think>"]
    for _ in range(5000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        assert block_count(text) == belt_count(text)
        assert block_end_offsets(text) == [e for _, e in block_ending_separators(text)]
        kind = classify_stop(
            text, finish_reason="stop", stop_reason="\n\n", hit_eos=False
        ).kind
        assert ends_on_block(text) is (kind == "block")
