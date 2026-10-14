"""Text views of structured assistant messages for scoring and consensus.

The original message is kept separately and returned unchanged to the client.
Tool-call IDs identify a particular generation, not the proposed action.
"""

import json
from typing import Any, Dict


def assistant_message_text(message: Dict[str, Any]) -> str:
    """Render a candidate for text scorers without discarding its tool arguments."""
    parts = [message.get("content") or ""]
    for call in message.get("tool_calls") or []:
        function = call["function"]
        parts.append(f"[tool_call: {function['name']}({function['arguments']})]")
    return "\n".join(part for part in parts if part)


def assistant_message_key(message: Dict[str, Any]) -> str:
    """Compare tool plans by ordered functions/arguments, ignoring sampled IDs."""
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
    return json.dumps({"tool_calls": plan}, sort_keys=True, ensure_ascii=False)
