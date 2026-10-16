"""Supervised fine-tuning (SFT) dataset schema detection and chat-template encoding.

Supported schemas (one per dataset, never mixed):

* ``instruction``: rows with ``instruction`` / ``input`` / ``output`` keys.
* ``messages``: rows with a ``messages`` (or ``turns``) list of ``{"role", "content"}``
  dicts, roles in ``system`` / ``user`` / ``assistant``.

Both schemas are rendered with the tokenizer chat template (falling back to ChatML when
the tokenizer has none) so that training and inference share the same prompt format.
Only ``assistant`` turns are supervised; ``system`` / ``user`` tokens are masked to -100.
Preference datasets (``chosen`` / ``rejected``) are rejected for the SFT path.
"""

from typing import Any, Dict, List, Optional

IGNORE_INDEX = -100

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

VALID_ROLES = ("system", "user", "assistant")
_INSTRUCTION_KEYS = ("instruction", "output")
_MESSAGE_KEYS = ("messages", "turns")
_PREFERENCE_KEYS = ("chosen", "rejected")


class SFTSchemaError(ValueError):
    """Raised for unsupported, ambiguous, or inconsistent SFT dataset schemas."""


class ChatTemplateError(ValueError):
    """Raised when the tokenizer chat template is incompatible with prefix-based masking."""


def detect_schema(example: Dict[str, Any]) -> str:
    """Return ``"instruction"`` or ``"messages"`` for a single example, or raise ``SFTSchemaError``."""
    if not isinstance(example, dict):
        raise SFTSchemaError(f"Dataset row must be an object, got {type(example).__name__}.")
    keys = {k for k, v in example.items() if v is not None}
    if any(k in keys for k in _PREFERENCE_KEYS):
        raise SFTSchemaError(
            "Preference dataset detected (chosen/rejected): use a DPO backend, not the SFT path."
        )
    has_msgs = any(k in keys for k in _MESSAGE_KEYS)
    has_instr = all(k in keys for k in _INSTRUCTION_KEYS)
    if has_msgs and (has_instr or "output" in keys):
        raise SFTSchemaError("Ambiguous row: contains both 'messages'/'turns' and instruction/output fields.")
    if has_msgs:
        if "messages" in keys and "turns" in keys:
            raise SFTSchemaError("Ambiguous row: contains both 'messages' and 'turns'.")
        return "messages"
    if has_instr:
        return "instruction"
    raise SFTSchemaError(
        f"Unrecognized SFT row schema with keys {sorted(keys)}; expected instruction/input/output or messages."
    )


def detect_dataset_schema(rows) -> str:
    """Validate that every row shares the same schema and return it."""
    schema: Optional[str] = None
    count = 0
    for idx, row in enumerate(rows):
        s = detect_schema(row)
        if schema is None:
            schema = s
        elif s != schema:
            raise SFTSchemaError(f"Mixed dataset schemas: row {idx} is '{s}' but earlier rows are '{schema}'.")
        count += 1
    if schema is None or count == 0:
        raise SFTSchemaError("Dataset is empty.")
    return schema


def to_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """Normalize a row (either schema) to a validated list of chat messages."""
    schema = detect_schema(example)
    if schema == "instruction":
        instruction = str(example.get("instruction") or "")
        extra = str(example.get("input") or "")
        user = instruction if not extra.strip() else f"{instruction}\n\n{extra}"
        msgs = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(example.get("output") or "")},
        ]
    else:
        raw = example.get("messages") if example.get("messages") is not None else example.get("turns")
        if not isinstance(raw, list) or not raw:
            raise SFTSchemaError("'messages'/'turns' must be a non-empty list.")
        msgs = []
        for m in raw:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise SFTSchemaError("Each message must be an object with 'role' and 'content'.")
            role = str(m["role"]).strip().lower()
            if role not in VALID_ROLES:
                raise SFTSchemaError(f"Unsupported message role '{m['role']}'.")
            msgs.append({"role": role, "content": str(m["content"] if m["content"] is not None else "")})
    if not any(m["role"] == "assistant" for m in msgs):
        raise SFTSchemaError("Conversation has no assistant turn to supervise.")
    return msgs


def ensure_chat_template(tokenizer) -> bool:
    """Install the ChatML fallback template if the tokenizer has none. Returns True if fallback was installed."""
    if getattr(tokenizer, "chat_template", None):
        return False
    tokenizer.chat_template = CHATML_TEMPLATE
    return True


def _render_ids(tokenizer, messages, add_generation_prompt: bool) -> List[int]:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def encode_messages(tokenizer, messages: List[Dict[str, str]], max_length: int) -> Dict[str, List[int]]:
    """Encode a conversation supervising only assistant turns (system/user masked with -100)."""
    full_ids = _render_ids(tokenizer, messages, add_generation_prompt=False)
    labels = [IGNORE_INDEX] * len(full_ids)

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix_ids = _render_ids(tokenizer, messages[:i], add_generation_prompt=True) if i > 0 else []
        through_ids = _render_ids(tokenizer, messages[: i + 1], add_generation_prompt=False)
        if full_ids[: len(prefix_ids)] != prefix_ids or full_ids[: len(through_ids)] != through_ids:
            raise ChatTemplateError(
                "Chat template is not prefix-consistent; cannot compute assistant-only label mask."
            )
        for j in range(len(prefix_ids), len(through_ids)):
            labels[j] = full_ids[j]

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and (not full_ids or full_ids[-1] != eos_id) and messages[-1]["role"] == "assistant":
        full_ids.append(eos_id)
        labels.append(eos_id)

    full_ids = full_ids[:max_length]
    labels = labels[:max_length]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def encode_example(tokenizer, example: Dict[str, Any], max_length: int) -> Dict[str, List[int]]:
    """Single encoder shared by train/val/test for both schemas."""
    return encode_messages(tokenizer, to_messages(example), max_length)


def split_for_generation(example: Dict[str, Any]):
    """Return (prompt_messages, expected_answer) using the last assistant turn as target."""
    msgs = to_messages(example)
    last = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    return msgs[:last], msgs[last]["content"]
