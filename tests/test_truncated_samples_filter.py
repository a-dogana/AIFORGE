import pytest

from aiforge.core.config import ExperimentConfig
from aiforge.llm.engine import LLMEngine
from aiforge.llm.sft_data import (
    IGNORE_INDEX,
    filter_supervised_examples,
    has_supervised_target_tokens,
)
from tests.test_sft_conversations import _SimpleChatTokenizer


def test_has_supervised_target_tokens_causal_shift():
    """Causal shift requires at least one supervised token at index >= 1 (labels[1:] != -100)."""
    assert has_supervised_target_tokens([]) is False
    assert has_supervised_target_tokens([42]) is False
    # Only index 0 is unmasked -> after causal shift labels[1:] is all -100 -> invalid!
    assert has_supervised_target_tokens([42, IGNORE_INDEX, IGNORE_INDEX]) is False
    # All -100 -> invalid
    assert has_supervised_target_tokens([IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]) is False
    # Supervised token at index >= 1 -> valid
    assert has_supervised_target_tokens([IGNORE_INDEX, 42, IGNORE_INDEX]) is True


def test_truncated_long_prompts_are_discarded_before_split():
    """Examples where prompt exceeds max_seq_length have all -100 labels and are discarded with accurate stats."""
    cfg = ExperimentConfig(
        experiment_name="TruncFilter",
        domain="llm",
        task="generation",
        device="cpu",
        data={"type": "jsonl", "csv_config": {"path": "dummy.jsonl", "target_column": "output"}},
        architecture={
            "type": "llm",
            "llm_config": {"model_id": "gpt2", "max_seq_length": 60, "lora_enabled": True},
        },
    )
    engine = LLMEngine(cfg)
    engine.tokenizer = _SimpleChatTokenizer()

    raw_rows = [
        # 4 valid short examples
        {"instruction": "Q1", "input": "", "output": "A1"},
        {"instruction": "Q2", "input": "", "output": "A2"},
        {"instruction": "Q3", "input": "", "output": "A3"},
        {"instruction": "Q4", "input": "", "output": "A4"},
        # 2 over-length prompt examples whose assistant response is completely truncated beyond max_seq_length=60
        {"instruction": "X" * 200, "input": "", "output": "NeverReached1"},
        {"instruction": "Y" * 200, "input": "", "output": "NeverReached2"},
    ]

    schema, valid_rows, stats = engine._encode_sft_dataset(raw_rows)
    assert schema == "instruction"
    assert stats == {"initial": 6, "valid": 4, "discarded": 2}
    assert len(valid_rows) == 4
    for row in valid_rows:
        assert has_supervised_target_tokens(row["labels"]) is True


def test_insufficient_valid_examples_after_truncation_raises_before_training():
    """When too many examples are truncated leaving < 3 valid rows, _encode_sft_dataset fails preventively."""
    cfg = ExperimentConfig(
        experiment_name="TruncFail",
        domain="llm",
        task="generation",
        device="cpu",
        data={"type": "jsonl", "csv_config": {"path": "dummy.jsonl", "target_column": "output"}},
        architecture={
            "type": "llm",
            "llm_config": {"model_id": "gpt2", "max_seq_length": 45, "lora_enabled": True},
        },
    )
    engine = LLMEngine(cfg)
    engine.tokenizer = _SimpleChatTokenizer()

    raw_rows = [
        {"instruction": "Short", "input": "", "output": "Ok"},
        {"instruction": "L" * 200, "input": "", "output": "Truncated1"},
        {"instruction": "M" * 200, "input": "", "output": "Truncated2"},
        {"instruction": "N" * 200, "input": "", "output": "Truncated3"},
    ]

    with pytest.raises(ValueError, match="Insufficient valid SFT examples"):
        engine._encode_sft_dataset(raw_rows)
