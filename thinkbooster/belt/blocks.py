"""BELT reasoning blocks, as the probe was trained and queried on them.

Copied from BELT `rl_controller/rollout_contract.py` and `rollout_runner.py`
(pinned fc9b067). A block ends at a blank line; generation stops on "\\n\\n"
with the stop string kept in the text; the head sees an end marker at EVERY
block end, placed after the separator; a policy is queried only when the text
ends on a block end, and never at block 1.
"""

import re

BLOCK_SEPARATOR = re.compile(r"\n[ \t]*\n")
BLOCK_STOP_STRING = "\n\n"
CLOSE_TAG = "</think>"


def block_end_offsets(text):
    """Character offsets just past every separator that closes a block.

    A separator closes a block only when non-whitespace text precedes it since
    the previous separator, so "\\n\\n\\n\\n" is one boundary, not two.
    """
    out, position = [], 0
    for match in BLOCK_SEPARATOR.finditer(text):
        if text[position : match.start()].strip():
            out.append(match.end())
        position = match.end()
    return out


def block_count(text):
    """How many blocks the separator regex sees: the 1-based block index of
    a prefix that ends on a block end."""
    spans, position = 0, 0
    for match in BLOCK_SEPARATOR.finditer(text):
        if text[position : match.start()].strip():
            spans += 1
        position = match.end()
    return spans + bool(text[position:].strip())


def ends_on_block(text):
    """True when a policy may be queried after this text.

    `classify_stop` returning "block", verbatim: thinking still open, only
    whitespace after the last separator, and non-whitespace text between it
    and the separator before. A stop that adds only a blank line is "partial".
    """
    if CLOSE_TAG in text:
        return False
    matches = list(BLOCK_SEPARATOR.finditer(text))
    if matches and not text[matches[-1].end() :].strip():
        before = text[: matches[-1].start()]
        prior = list(BLOCK_SEPARATOR.finditer(before))
        start = prior[-1].end() if prior else 0
        return bool(before[start:].strip())
    return False
