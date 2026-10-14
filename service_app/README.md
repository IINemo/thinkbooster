# ThinkBooster Service

OpenAI-compatible REST API for Test-Time Scaling strategies.

## Overview

This service exposes TTS strategies through an **OpenAI-compatible API**. You can use it as a drop-in replacement for OpenAI's API, leveraging advanced test-time scaling with your existing OpenAI SDK code.

## Supported Strategies

| Strategy | Backend | Description |
|---|---|---|
| `self_consistency` | OpenRouter / OpenAI API | Majority voting over multiple reasoning paths |
| `offline_bon` | Local vLLM | Offline Best-of-N: generate N trajectories, pick best |
| `online_bon` | Local vLLM | Online Best-of-N: step-level candidate selection |
| `beam_search` | Local vLLM | Beam search over reasoning steps |

## Quick Start

### Setup

```bash
# From repository root
pip install -e ".[service]"
export OPENROUTER_API_KEY="your-key"
python service_app/main.py
```

The service starts on `http://localhost:8001`. Open http://localhost:8001/docs for interactive API documentation.

## Usage

### Self-Consistency (API models)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="your-openrouter-key"
)

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Reason step by step, put answer in \\boxed{}."},
        {"role": "user", "content": "What is 15 * 7?"}
    ],
    extra_body={
        "tts_strategy": "self_consistency",
        "num_paths": 5
    }
)

print(response.choices[0].message.content)
print(response.choices[0].tts_metadata)  # consensus_score, answer_distribution
```

### Tool calling (OpenAI-compatible API backends)

The chat endpoint accepts `tools`, `tool_choice` (`auto`, `none`, `required`, or
an explicit function choice), and `parallel_tool_calls`. It preserves assistant
`tool_calls`, nullable assistant `content`, and tool-result messages with their
`tool_call_id`. Function definitions support `description`, `parameters`, and
`strict`. Developer messages are also accepted.

Configure `OPENAI_API_KEY` on the server for this example, or pass an upstream key
as `tts_api_key` in `extra_body`:

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="unused")
messages = [{"role": "user", "content": "What is the temperature in Dubai?"}]
tools = [{
    "type": "function",
    "function": {
        "name": "get_temperature",
        "description": "Get the current temperature for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
options = {"provider": "openai", "tts_strategy": "self_consistency", "num_paths": 3}
response = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, tools=tools,
    tool_choice="required", parallel_tool_calls=False, extra_body=options,
)
assistant = response.choices[0].message
messages.append(assistant.model_dump(exclude_none=True))
for call in assistant.tool_calls or []:
    # Execute the selected function in your application. This sample result is illustrative.
    result = {"city": json.loads(call.function.arguments)["city"], "celsius": 30}
    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
response = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, tools=tools,
    tool_choice="none", extra_body=options,
)
print(response.choices[0].message.content)
```

Tool-enabled requests use complete assistant turns as candidates across all four
strategies. Step boundaries and assistant prefill are disabled for these turns,
so JSON arguments are never split at reasoning markers. Self-consistency votes
on function names and parsed arguments, ignoring call IDs and JSON key order;
text-only replies on these requests use exact-text voting. Best-of-N and beam
search score the complete turn. Only the selected message is returned, with its
original call IDs, arguments, and upstream `finish_reason`. ThinkBooster never
executes the tools; the client executes the selected calls and sends results back.

With `stream=True`, tool-enabled requests return OpenAI `chat.completion.chunk`
events after selection, followed by `[DONE]`. Tool-call deltas have an `index` for
each parallel call. Unselected candidates are not streamed. Requests without tool
fields or tool history retain the existing progress-event streaming format.

Use `model_base_url` for another OpenAI-compatible backend, including a vLLM HTTP
server configured for tool calling. The in-process `provider="vllm"` backend does
not support this protocol and returns HTTP 400 for tool-enabled requests without
`model_base_url`. The upstream model must support the requested tool options.

Confidence scorers still depend on upstream logprobs. Providers often omit
logprobs for tool calls, so those calls receive neutral confidence scores; this
change does not add a tool-correctness verifier. Self-consistency does not require
logprobs. Each candidate consumes a separate upstream completion, and streaming
begins only after candidate selection.

### vLLM Strategies (local model)

Requires `VLLM_MODEL_PATH` to be set.

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[...],
    extra_body={
        "tts_strategy": "offline_bon",
        "tts_num_trajectories": 8,
        "tts_scorer": "entropy",        # entropy, perplexity, sequence_prob, prm
        "tts_score_aggregation": "min",  # min, mean, max, product, last
    }
)
```

### With cURL

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [
      {"role": "system", "content": "Reason step by step, put answer in \\boxed{}."},
      {"role": "user", "content": "What is 15 * 7?"}
    ],
    "tts_strategy": "self_consistency",
    "num_paths": 5
  }'
```

## API Reference

### GET /debugger

Interactive visual debugger demo for reasoning trajectories.

- Compare multiple TTS-style strategies under the same prompt and budget
- Run the full strategy x scorer matrix for a user-provided single sample
- Inspect per-step decisions (escalate/stop/prune/select)
- View confidence and uncertainty signals behind each decision
- Drill into sampled candidates and tree expansions
- Includes a local prototype example (`prototype_local_demo`) when backend APIs are unavailable

Custom input modes in the UI:
- `Single example`: enter one question + gold answer
- Optional shared prompt/instruction can be applied to the custom sample
- Model configuration inputs are available: `provider` (`openai` or `openrouter`), `model-id`, and `API key`
- Current limitation: these model credentials are captured in the debugger UI for future backend integration, but backend execution still uses keys from `.env`

Prototype-only usage (no running backend):

```bash
# Open directly in a browser
service_app/static/debugger/index.html
```

Cached prototype scenarios are stored in:

```bash
service_app/static/debugger/cached_examples.json
```

### POST /v1/chat/completions

**Standard OpenAI Parameters:**
- `model` (string): Model to use (e.g., `openai/gpt-4o-mini`)
- `messages` (array): Chat messages
- `temperature` (float): Sampling temperature (0-2, default: 0.7)
- `max_tokens` (int): Maximum tokens (default: 4096)

**TTS Parameters (via `extra_body`):**
- `tts_strategy` (string): `self_consistency`, `offline_bon`, `online_bon`, `beam_search`
- `provider` (string): `openrouter`, `openai`, or `vllm` (auto-detected from strategy)
- `num_paths` (int): Reasoning paths for self_consistency (default: 5)
- `tts_scorer` (string): `entropy`, `perplexity`, `sequence_prob`, `prm` (vLLM strategies)
- `tts_num_trajectories` (int): Trajectories for offline_bon (default: 8)
- `tts_candidates_per_step` (int): Candidates per step for online_bon/beam_search (default: 4)
- `tts_beam_size` (int): Beam size for beam_search (default: 4)
- `tts_max_steps` (int): Max reasoning steps (default: 100)
- `tts_score_aggregation` (string): `min`, `mean`, `max`, `product`, `last`
- `tts_window_size` (int): Scoring window size (optional)

### GET /v1/models

List available models.

### GET /health

Health check endpoint.

## Configuration

### Environment Variables

See `service_app/.env.example` for all options:

```bash
# API Keys
OPENROUTER_API_KEY=your-key    # Required for self_consistency
OPENAI_API_KEY=your-key        # Optional

# Server
PORT=8001
HOST=0.0.0.0

# vLLM Backend (for offline_bon, online_bon, beam_search)
VLLM_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
VLLM_MAX_MODEL_LEN=32000
VLLM_GPU_MEMORY_UTILIZATION=0.5

# PRM Scorer (optional)
PRM_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
PRM_USE_VLLM=true
PRM_GPU_MEMORY_UTILIZATION=0.3
```

## Project Structure

```
service_app/
├── api/
│   ├── routes/
│   │   ├── chat.py            # /v1/chat/completions
│   │   └── models.py          # /v1/models
│   └── models/
│       └── openai_compat.py   # OpenAI-compatible Pydantic schemas
├── core/
│   ├── config.py              # Configuration (pydantic-settings)
│   ├── strategy_manager.py    # Strategy creation and lifecycle
│   ├── prm_scorer_factory.py  # PRM scorer lazy initialization
│   └── logging_config.py      # Logging setup
├── main.py                    # FastAPI app entry point
├── Dockerfile
└── .env.example
```
