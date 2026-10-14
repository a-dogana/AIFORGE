import os
import pytest
import json
from unittest.mock import MagicMock
from aiforge.deployment.gguf_export import GGUFExporter, SUPPORTED_GGUF_QUANTIZATIONS


def test_gguf_supported_quantizations():
    assert "Q4_K_M" in SUPPORTED_GGUF_QUANTIZATIONS
    assert "Q8_0" in SUPPORTED_GGUF_QUANTIZATIONS
    assert "F16" in SUPPORTED_GGUF_QUANTIZATIONS


def test_gguf_export_non_llm_rejection(tmp_path):
    mock_reg = MagicMock()
    mock_reg.get_experiment_record.return_value = {
        "id": "exp_vision",
        "config": {"architecture": {"type": "cnn"}},
        "checkpoint_path": str(tmp_path)
    }

    exporter = GGUFExporter(mock_reg)
    with pytest.raises(ValueError, match="Only LLM models can be exported"):
        exporter.export("exp_vision", quantization="Q4_K_M")


def test_gguf_export_invalid_quantization(tmp_path):
    mock_reg = MagicMock()
    exporter = GGUFExporter(mock_reg)
    with pytest.raises(ValueError, match="Unsupported quantization"):
        exporter.export("exp_llm", quantization="INVALID_QUANT")


def test_gguf_export_missing_checkpoint():
    mock_reg = MagicMock()
    mock_reg.get_experiment_record.return_value = {
        "id": "exp_llm",
        "config": {"architecture": {"type": "llm"}},
        "checkpoint_path": "/nonexistent/checkpoint/path"
    }

    exporter = GGUFExporter(mock_reg)
    with pytest.raises(FileNotFoundError):
        exporter.export("exp_llm", quantization="Q4_K_M")


def test_gguf_export_success(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    mock_reg = MagicMock()
    mock_reg.get_experiment_record.return_value = {
        "id": "qwen_exp",
        "config": {
            "architecture": {
                "type": "llm",
                "llm_config": {
                    "model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                    "max_seq_length": 2048
                }
            },
            "backend_args": {"chat_template": "chatml"}
        },
        "checkpoint_path": str(checkpoint_dir)
    }

    exporter = GGUFExporter(mock_reg)
    res = exporter.export("qwen_exp", quantization="Q4_K_M")

    assert res["status"] == "success"
    assert res["quantization"] == "Q4_K_M"
    assert os.path.exists(res["meta_path"])
    assert os.path.exists(res["script_path"])
    assert os.path.exists(res["modelfile_path"])

    with open(res["meta_path"], "r", encoding="utf-8") as f:
        meta = json.load(f)
        assert meta["format"] == "GGUF"
        assert meta["architecture"] == "llama"
        assert meta["quantization"] == "Q4_K_M"
        assert meta["context_length"] == 2048
