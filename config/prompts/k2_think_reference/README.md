# K2-Think UHead training prompts — REFERENCE ONLY

These two files are the exact completion prefixes the K2-Think-V2 UHead was
trained on (provided by the UHead author). They are kept here **for provenance /
reference only** — do **not** set `dataset.prompt_file` to them.

They are full, already-chat-templated strings (literal `<|im_start|>` tokens,
ending mid-assistant-turn at `<think_fast>` / `<think_faster>`). The runtime
always re-applies the chat template, so using them directly would double-wrap the
prompt.

To reproduce them faithfully at inference, the experiment configs instead use:

- `dataset.prompt_file: config/prompts/k2_think_answer.txt` — the user-message
  wrapper with the `{question}` placeholder, and
- `model.reasoning_effort: medium|low` — K2-Think-V2's chat template maps this to
  the `<think_fast>` / `<think_faster>` budget token.

The chat template applied to the wrapper + reasoning_effort yields a string byte-
identical to these references (with `{q}` filled by the dataset question).

| File | Budget token | reasoning_effort |
|------|--------------|------------------|
| `k2_think_medium_completion_prefix.txt` | `<think_fast>`   | medium |
| `k2_think_low_completion_prefix.txt`    | `<think_faster>` | low    |
