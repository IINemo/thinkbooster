"""OpenAI tool protocol round trips through real generators and strategies.

Only upstream HTTP and token counting are replaced; no provider credentials or
model downloads are needed. The service and OpenAI SDK parse real wire payloads.
"""

import copy
import json
import math
import threading

import httpx
import pytest
from openai import OpenAI

from service_app.core.config import settings
from thinkbooster.generators.api import StepCandidateGeneratorThroughAPI
from thinkbooster.models.tool_calls import assistant_message_key

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def tool_message(call_id="call_selected", arguments='{"path":"main.py"}'):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": arguments,
                },
            }
        ],
    }


@pytest.fixture
def upstream(monkeypatch):
    """Queue responses behind an actual OpenAI SDK MockTransport."""
    state = {"messages": [], "requests": [], "finish_reason": None, "probabilities": {}}
    clients = []
    lock = threading.Lock()

    def respond(request):
        payload = json.loads(request.content)
        with lock:
            state["requests"].append(payload)
            assert state["messages"], "Unexpected extra generation/continuation"
            message = copy.deepcopy(state["messages"].pop(0))
        reason = state["finish_reason"] or (
            "tool_calls" if message.get("tool_calls") else "stop"
        )
        logprobs = None
        probability = state["probabilities"].get(message.get("content"))
        if probability is not None:
            logprobs = {
                "content": [
                    {
                        "token": message["content"],
                        "logprob": math.log(probability),
                        "top_logprobs": [
                            {
                                "token": message["content"],
                                "logprob": math.log(probability),
                            },
                            {
                                "token": "alternative",
                                "logprob": math.log(1 - probability),
                            },
                        ],
                    }
                ]
            }
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-upstream",
                "object": "chat.completion",
                "created": 1,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": reason,
                        "logprobs": logprobs,
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            },
        )

    def client_factory(**kwargs):
        client = OpenAI(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond))
        )
        clients.append(client)
        return client

    def token_counter(self, model_name):
        self._tokenizer = None
        self._count_tokens = lambda text: len(text) // 4

    monkeypatch.setattr(
        "thinkbooster.models.blackboxmodel_with_streaming.openai.OpenAI", client_factory
    )
    monkeypatch.setattr(
        StepCandidateGeneratorThroughAPI, "_init_tokenizer", token_counter
    )
    # Exercise routing via server configuration, without a per-request key.
    monkeypatch.setattr(settings, "openai_api_key", "test-only")
    monkeypatch.setattr(settings, "openrouter_api_key", "test-only")
    yield state
    for client in clients:
        client.close()


def body(**kwargs):
    return {
        "model": "test-model",
        "provider": "openai",
        "tools": TOOLS,
        "messages": [{"role": "user", "content": "Read main.py"}],
        "num_paths": 1,
        "tts_num_trajectories": 1,
        "tts_candidates_per_step": 1,
        "tts_beam_size": 1,
        "tts_max_steps": 3,
        "max_tokens": 4096,
        **kwargs,
    }


@pytest.mark.parametrize(
    "strategy", ["self_consistency", "offline_bon", "online_bon", "beam_search"]
)
def test_tool_result_round_trip_all_strategies(test_client, upstream, strategy):
    # A step marker inside JSON must not cut off tool arguments.
    selected = tool_message(arguments='{"path":"First, main.py\\n<end of response>"}')
    upstream["messages"] = [selected]
    request = body(
        tts_strategy=strategy, tool_choice="required", parallel_tool_calls=False
    )
    response = test_client.post("/v1/chat/completions", json=request)
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"] == selected["tool_calls"]
    sent = upstream["requests"][0]
    assert sent["tools"] == TOOLS
    assert sent["tool_choice"] == "required"
    assert sent["parallel_tool_calls"] is False
    assert sent["stream"] is False
    assert sent["stop"] is None
    assert sent["max_tokens"] == 4096

    # The client executes the chosen tool and returns its result on the next turn.
    request["messages"] += [
        selected,
        {"role": "tool", "tool_call_id": "call_selected", "content": "print('hello')"},
    ]
    request["tool_choice"] = "none"
    upstream["messages"] = [{"role": "assistant", "content": "The file prints hello."}]
    response = test_client.post("/v1/chat/completions", json=request)
    assert response.status_code == 200, response.text
    assert (
        response.json()["choices"][0]["message"]["content"] == "The file prints hello."
    )
    assert upstream["requests"][-1]["messages"] == request["messages"]
    assert len(upstream["requests"]) == 2


def test_majority_votes_on_arguments_not_call_ids_or_numbers(test_client, upstream):
    winner = tool_message("one", '{"path": "file2.py", "line": 10}')
    equivalent = tool_message("two", '{"line":10,"path":"file2.py"}')
    loser = tool_message("three", '{"path":"other.py","line":10}')
    upstream["messages"] = [loser, winner, equivalent]
    response = test_client.post("/v1/chat/completions", json=body(num_paths=3))
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["message"]["tool_calls"] in [
        winner["tool_calls"],
        equivalent["tool_calls"],
    ]
    assert choice["tts_metadata"]["results"]["consensus_score"] == pytest.approx(2 / 3)
    assert len(upstream["requests"]) == 3


@pytest.mark.parametrize("model", ["test-model", "gpt-4o"])
def test_openai_sdk_stream_parallel_calls(test_client, upstream, model):
    selected = tool_message()
    selected["tool_calls"] += tool_message("second", '{"path":"tests.py"}')[
        "tool_calls"
    ]
    upstream["messages"] = [selected]
    with OpenAI(
        base_url="http://testserver/v1",
        api_key="unused",
        http_client=type(test_client)(test_client.app),
    ) as sdk:
        stream = sdk.chat.completions.create(
            model=model,
            messages=body()["messages"],
            tools=TOOLS,
            stream=True,
            tool_choice={"type": "function", "function": {"name": "read_file"}},
            parallel_tool_calls=True,
            extra_body={"provider": "openai", "num_paths": 1},
        )
        chunks = list(stream)
    assert len(chunks) == 2
    calls = chunks[0].choices[0].delta.tool_calls
    assert [call.model_dump(exclude={"index"}) for call in calls] == selected[
        "tool_calls"
    ]
    assert [call.index for call in calls] == [0, 1]
    assert chunks[1].choices[0].finish_reason == "tool_calls"
    assert upstream["requests"][0]["tool_choice"]["function"]["name"] == "read_file"


def test_tool_history_without_new_tools_is_preserved(test_client, upstream):
    request = body()
    del request["tools"]
    request["messages"] += [
        tool_message(),
        {"role": "tool", "tool_call_id": "call_selected", "content": "done"},
    ]
    upstream["messages"] = [{"role": "assistant", "content": "Done"}]
    response = test_client.post("/v1/chat/completions", json=request)
    assert response.status_code == 200, response.text
    assert upstream["requests"][0]["messages"] == request["messages"]
    for field in ("tools", "tool_choice", "parallel_tool_calls"):
        assert field not in upstream["requests"][0]


@pytest.mark.parametrize("reason", ["length", "content_filter"])
def test_upstream_finish_reason_is_preserved(test_client, upstream, reason):
    upstream["messages"] = [tool_message(arguments='{"path":')]
    upstream["finish_reason"] = reason
    response = test_client.post("/v1/chat/completions", json=body())
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["finish_reason"] == reason
    assert len(upstream["requests"]) == 1


@pytest.mark.parametrize(
    "message",
    [
        {"role": "tool", "content": "result"},
        {"role": "user", "content": None},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "user", "content": "hi", "tool_calls": tool_message()["tool_calls"]},
    ],
)
def test_invalid_tool_history_rejected(test_client, message):
    response = test_client.post("/v1/chat/completions", json=body(messages=[message]))
    assert response.status_code == 422


def test_local_vllm_tool_request_fails_before_stream_starts(test_client):
    response = test_client.post(
        "/v1/chat/completions", json=body(provider="vllm", stream=True)
    )
    assert response.status_code == 400
    assert "model_base_url" in response.text


def test_tool_consensus_retains_order_and_handles_invalid_json():
    first = tool_message(arguments="not json")
    assert assistant_message_key(first) == assistant_message_key(
        tool_message("new-id", "not json")
    )
    second = tool_message(arguments='{"path":"second.py"}')
    ordered = {"tool_calls": first["tool_calls"] + second["tool_calls"]}
    reversed_calls = {"tool_calls": list(reversed(ordered["tool_calls"]))}
    assert assistant_message_key(ordered) != assistant_message_key(reversed_calls)
    assert assistant_message_key(first) != assistant_message_key(
        {"content": "not json"}
    )


@pytest.mark.parametrize("strategy", ["offline_bon", "online_bon", "beam_search"])
def test_scoring_preserves_the_winning_message(test_client, upstream, strategy):
    winner = tool_message("winner", '{"path":"right.py"}')
    winner["content"] = "Read the right file."
    loser = tool_message("loser", '{"path":"wrong.py"}')
    loser["content"] = "Read the wrong file."
    upstream["messages"] = [loser, winner]
    upstream["probabilities"] = {loser["content"]: 0.5, winner["content"]: 0.99}
    response = test_client.post(
        "/v1/chat/completions",
        json=body(
            model="gpt-4o",
            tts_strategy=strategy,
            tts_num_trajectories=2,
            tts_candidates_per_step=2,
            tts_beam_size=2,
        ),
    )
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["message"]["content"] == winner["content"]
    assert choice["message"]["tool_calls"] == winner["tool_calls"]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["tts_metadata"]["validity_scores"][0] > 0.9
    assert len(upstream["requests"]) == 2
    assert all(request["logprobs"] for request in upstream["requests"])


def test_tool_stream_ends_with_done(test_client, upstream):
    upstream["messages"] = [tool_message()]
    response = test_client.post("/v1/chat/completions", json=body(stream=True))
    assert response.text.endswith("data: [DONE]\n\n")
    assert '"type": "progress"' not in response.text
    assert '"type": "complete"' not in response.text
