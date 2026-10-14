"""Text views of structured assistant messages for scoring and consensus.

The original message is kept separately and returned unchanged to the client.
Tool-call IDs identify a particular generation, not the proposed action.
"""

import json
from typing import Any, Dict, Optional

# Upstream stopped before the turn was finished; any tool arguments are partial.
_CUT_OFF_FINISH_REASONS = {"length", "content_filter"}


def assistant_message_text(message: Dict[str, Any]) -> str:
    """Render a candidate for text scorers without discarding its tool arguments."""
    parts = [message.get("content") or ""]
    for call in message.get("tool_calls") or []:
        function = call["function"]
        parts.append(f"[tool_call: {function['name']}({function['arguments']})]")
    return "\n".join(part for part in parts if part)


def is_well_formed_turn(
    message: Dict[str, Any], finish_reason: Optional[str] = None
) -> bool:
    """A turn a client can act on: not cut off, and every argument string parses."""
    if finish_reason in _CUT_OFF_FINISH_REASONS:
        return False
    for call in message.get("tool_calls") or []:
        arguments = call["function"]["arguments"]
        # Some providers send "" for functions that take no parameters.
        if not arguments or not arguments.strip():
            continue
        try:
            json.loads(arguments)
        except (ValueError, TypeError):
            return False
    return True


def assistant_message_key(message: Dict[str, Any]) -> str:
    """Compare tool plans by functions/arguments, ignoring sampled IDs.

    Calls in one turn are issued together, before any result is known, so their
    order carries no meaning and is normalised away.
    """
    calls = message.get("tool_calls") or []
    if not calls:
        return json.dumps({"content": message.get("content")}, ensure_ascii=False)
    plan = []
    for call in calls:
        function = call["function"]
        arguments = function["arguments"]
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError):
            pass
        plan.append({"name": function["name"], "arguments": arguments})
    plan.sort(key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    return json.dumps({"tool_calls": plan}, sort_keys=True, ensure_ascii=False)
