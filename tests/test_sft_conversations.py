import pytest
import torch

from aiforge.core.config import ExperimentConfig
from aiforge.llm.engine import LLMEngine
from aiforge.llm.sft_data import (
    CHATML_TEMPLATE,
    IGNORE_INDEX,
    ChatTemplateError,
    SFTSchemaError,
    detect_dataset_schema,
    encode_example,
    ensure_chat_template,
    split_for_generation,
)


class _SimpleChatTokenizer:
    """Deterministic word/char tokenizer supporting Jinja-like ChatML rendering for fast unit tests."""

    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self, chat_template=None):
        self.chat_template = chat_template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if self.chat_template == "BROKEN_NON_PREFIX":
            # Intentionally prepend total message count so prefixes do not match full render
            header = f"[TOTAL={len(messages)}]\n"
            body = "".join(f"<{m['role']}>{m['content']}</{m['role']}>\n" for m in messages)
            return header + body
        out = []
        for m in messages:
            out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
        rendered = "".join(out)
        if tokenize:
            return self(rendered)["input_ids"]
        return rendered

    def __call__(self, text, add_special_tokens=False, return_tensors=None, **kwargs):
        ids = [ord(c) % 250 + 3 for c in text]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, token_ids, skip_special_tokens=True):
        return "decoded_response"


def test_single_turn_instruction_and_multi_turn_messages_masking():
    """Both instruction/input/output and multi-turn messages supervise only assistant turns."""
    tok = _SimpleChatTokenizer()
    ensure_chat_template(tok)
    assert tok.chat_template == CHATML_TEMPLATE

    # 1. Single-turn instruction/input/output
    ex_inst = {"instruction": "Explain recursion", "input": "in Python", "output": "A function calling itself."}
    assert detect_dataset_schema([ex_inst]) == "instruction"
    enc_inst = encode_example(tok, ex_inst, max_length=512)
    supervised_inst = [t for t, l in zip(enc_inst["input_ids"], enc_inst["labels"]) if l != IGNORE_INDEX]
    masked_inst = [t for t, l in zip(enc_inst["input_ids"], enc_inst["labels"]) if l == IGNORE_INDEX]
    assert len(supervised_inst) > 0
    assert len(masked_inst) > 0

    # 2. Multi-turn messages: system + user1 + assistant1 + user2 + assistant2
    ex_multi = {
        "messages": [
            {"role": "system", "content": "You are a Dart architect."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"},
        ]
    }
    assert detect_dataset_schema([ex_multi]) == "messages"
    enc_multi = encode_example(tok, ex_multi, max_length=1024)
    labels = enc_multi["labels"]

    # Verify there are two distinct contiguous spans of supervised tokens (one for each assistant turn)
    spans = 0
    in_span = False
    for val in labels:
        if val != IGNORE_INDEX and not in_span:
            spans += 1
            in_span = True
        elif val == IGNORE_INDEX:
            in_span = False
    assert spans == 2

    prompt_msgs, expected = split_for_generation(ex_multi)
    assert len(prompt_msgs) == 4
    assert expected == "Second answer"


def test_sft_rejects_dpo_ambiguous_mixed_and_broken_templates():
    """DPO preference rows, ambiguous rows, mixed datasets, and non-prefix templates fail explicitly."""
    # DPO preference dataset (chosen / rejected)
    with pytest.raises(SFTSchemaError, match="DPO"):
        detect_dataset_schema([{"prompt": "hi", "chosen": "hello", "rejected": "bye"}])

    # Ambiguous row containing both messages and instruction/output
    with pytest.raises(SFTSchemaError, match="Ambiguous"):
        detect_dataset_schema(
            [
                {
                    "instruction": "inst",
                    "output": "out",
                    "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
                }
            ]
        )

    # Mixed rows across dataset
    with pytest.raises(SFTSchemaError, match="Mixed dataset schemas"):
        detect_dataset_schema(
            [
                {"instruction": "inst", "input": "", "output": "out"},
                {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]},
            ]
        )

    # Non-prefix-consistent chat template
    bad_tok = _SimpleChatTokenizer(chat_template="BROKEN_NON_PREFIX")
    with pytest.raises(ChatTemplateError, match="prefix-consistent"):
        encode_example(
            bad_tok,
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
            max_length=512,
        )


def test_llm_engine_generate_applies_chat_template():
    """LLMEngine.generate wraps plain user prompts with the tokenizer's chat template when present."""
    cfg = ExperimentConfig(
        experiment_name="ChatGen",
        domain="llm",
        task="generation",
        device="cpu",
        data={"type": "jsonl", "csv_config": {"path": "dummy.jsonl", "target_column": "output"}},
        architecture={"type": "llm", "llm_config": {"model_id": "gpt2"}},
    )
    engine = LLMEngine(cfg)
    tok = _SimpleChatTokenizer(chat_template=CHATML_TEMPLATE)
    engine.tokenizer = tok

    recorded_input_lens = []

    class _DummyModel:
        def eval(self):
            return self

        def generate(self, input_ids, attention_mask=None, **kwargs):
            recorded_input_lens.append(input_ids.shape[1])
            return torch.cat([input_ids, torch.tensor([[10, 11, 12]], dtype=torch.long)], dim=1)

    engine.model = _DummyModel()
    raw_prompt = "Ciao come stai?"
    raw_len = len(tok(raw_prompt)["input_ids"])
    out = engine.generate(raw_prompt, max_new_tokens=10, temperature=0.0)
    assert out == "decoded_response"
    # ChatML wrapping adds <|im_start|>user...<|im_end|>\n<|im_start|>assistant\n so input length > raw_len
    assert recorded_input_lens[0] > raw_len
