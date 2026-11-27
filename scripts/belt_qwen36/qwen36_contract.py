"""Prompt contract for BELT on Qwen3.6-35B-A3B (FP8 checkpoint).

Mirrors BELT's ``olmo3_think_contract`` module (same functions, same user
turn), so BELT's generator, natural-answer fallback and trainer can select it
through ``rollout_contract``. ``apply_belt_patch.py`` copies this file into a
BELT checkout as ``rl_controller/qwen36_contract.py``.

Differences from OLMo-3, and why:

* Sampler: BELT's T = 0.6, top-p 0.95 (kept for both of its models), with
  Qwen's top-k 20. Qwen3.6's own default is T = 1.0; we keep BELT's.
* Budget: 16,384 thinking tokens as in BELT's OLMo recipe (``--budget``);
  ``MAX_TOKENS`` only bounds the prompt check below.
* The chat template opens ``<think>`` in the prompt, as OLMo-3's does.
* Transformers 5 returns a ``BatchEncoding`` from ``apply_chat_template``;
  ``list()`` of it would yield its keys, so ``input_ids`` is taken explicitly.
"""

from __future__ import annotations

import hashlib
import json

CONTRACT_ID = "qwen3.6-35b-a3b-fp8-native-thinking-v1"
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
REVISION = "c653bba46dfa06e9658dcc0b0dc87fe4685f14a2"
ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"
CHAT_TEMPLATE_SHA256 = (
    "1b8361565cc35d47551d342214cff341c0f1a9685ce529c7ce31de0ed3616248"
)
VOCAB_SIZE = 248320
HIDDEN_SIZE = 2048
NUM_HIDDEN_LAYERS = 40
MAX_TOKENS = 16384
MAX_MODEL_LEN = 20480
EOS_TOKEN_IDS = (248046, 248044)
PAD_TOKEN_ID = 248044
THINK_OPEN_ID = 248068
THINK_CLOSE_ID = 248069


def official_generation_config() -> dict:
    return {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "eos_token_id": list(EOS_TOKEN_IDS),
        "max_new_tokens": MAX_TOKENS,
    }


def resolved_generation_config() -> dict:
    return {
        **official_generation_config(),
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": MAX_TOKENS,
    }


def assert_official_generation_config(config) -> None:
    for field, expected in official_generation_config().items():
        actual = config.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"Qwen3.6 generation config field {field!r} must be "
                f"{expected!r}, got {actual!r}"
            )


def build_messages(question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "Answer the following question. Enclose your answer in "
                f"<answer></answer>.\n<Question>: {question}"
            ),
        }
    ]


def assert_tokenizer(tokenizer) -> None:
    template = getattr(tokenizer, "get_chat_template", None)
    template = (
        template() if callable(template) else getattr(tokenizer, "chat_template", None)
    )
    if not template:
        raise ValueError("the Qwen3.6 contract needs a tokenizer with a chat template")
    digest = hashlib.sha256(template.encode()).hexdigest()
    if digest != CHAT_TEMPLATE_SHA256:
        raise ValueError(
            f"chat template sha256 {digest} is not the pinned Qwen3.6 template "
            f"{CHAT_TEMPLATE_SHA256}: this tokenizer is not {MODEL}@{REVISION}"
        )


def render_prompt_ids(tokenizer, messages) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    ids = [int(t) for t in ids]
    if not ids:
        raise ValueError("empty prompt")
    # The generation prompt ends "<|im_start|>assistant\n<think>\n".
    if THINK_OPEN_ID not in ids[-3:]:
        raise ValueError("the Qwen3.6 prompt must end by opening <think>")
    return ids


def build_request_payload(tokenizer, model, question: str, seed: int):
    assert_tokenizer(tokenizer)
    messages = build_messages(question)
    prompt_ids = render_prompt_ids(tokenizer, messages)
    validate_prompt_budget(prompt_ids)
    payload = {
        "model": model,
        "messages": messages,
        "n": 1,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": MAX_TOKENS,
        "seed": seed,
        "add_generation_prompt": True,
        "return_token_ids": True,
        "stream": False,
    }
    return payload, prompt_ids


def validate_prompt_budget(prompt_ids) -> None:
    if len(prompt_ids) + MAX_TOKENS > MAX_MODEL_LEN:
        raise ValueError(
            f"{len(prompt_ids)} prompt tokens plus {MAX_TOKENS} completion tokens "
            f"exceed max_model_len={MAX_MODEL_LEN}"
        )


def identity() -> dict:
    # No feature layer here: generation does not depend on it, and the layer is
    # chosen after the chains exist (a sweep on the validation chains).
    return {
        "prompt_contract_id": CONTRACT_ID,
        "model_id": MODEL,
        "model_revision": REVISION,
        "architecture": ARCHITECTURE,
        "thinking_mode": "native_on",
        "think_open_in_prompt": True,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "generation": resolved_generation_config(),
        "max_model_len": MAX_MODEL_LEN,
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
    }


def make_run_fingerprint(dataset_fingerprint: str, chat_template: str) -> str:
    payload = {
        "contract": identity(),
        "dataset_fingerprint": dataset_fingerprint,
        "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
