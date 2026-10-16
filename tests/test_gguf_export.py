import os
import stat
import struct
import json
import pytest
from unittest.mock import MagicMock, patch
from aiforge.deployment.gguf_export import (
    GGUFExporter,
    SUPPORTED_GGUF_QUANTIZATIONS,
    validate_gguf_file,
)


def _write_dummy_gguf(path: str, version: int = 3, tensors: int = 4, kv: int = 2) -> None:
    header = struct.pack("<4sIQQ", b"GGUF", version, tensors, kv)
    with open(path, "wb") as f:
        f.write(header + b"\x00" * 32)


def test_gguf_supported_quantizations():
    assert "Q4_K_M" in SUPPORTED_GGUF_QUANTIZATIONS
    assert "Q8_0" in SUPPORTED_GGUF_QUANTIZATIONS
    assert "F16" in SUPPORTED_GGUF_QUANTIZATIONS


def test_validate_gguf_file_header(tmp_path):
    valid_path = tmp_path / "valid.gguf"
    _write_dummy_gguf(str(valid_path), version=3, tensors=10, kv=5)
    info = validate_gguf_file(str(valid_path))
    assert info["valid"] is True
    assert info["version"] == 3
    assert info["tensor_count"] == 10
    assert info["metadata_kv_count"] == 5

    # Missing file
    with pytest.raises(FileNotFoundError):
        validate_gguf_file(str(tmp_path / "missing.gguf"))

    # Truncated file
    short_path = tmp_path / "short.gguf"
    short_path.write_bytes(b"GGUF\x03")
    with pytest.raises(ValueError, match="smaller than the 24-byte header"):
        validate_gguf_file(str(short_path))

    # Wrong magic
    bad_magic = tmp_path / "bad_magic.gguf"
    bad_magic.write_bytes(struct.pack("<4sIQQ", b"BAD!", 3, 1, 1))
    with pytest.raises(ValueError, match="Invalid GGUF magic bytes"):
        validate_gguf_file(str(bad_magic))

    # Unsupported version
    bad_ver = tmp_path / "bad_ver.gguf"
    bad_ver.write_bytes(struct.pack("<4sIQQ", b"GGUF", 99, 1, 1))
    with pytest.raises(ValueError, match="Unsupported GGUF version"):
        validate_gguf_file(str(bad_ver))


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
    (checkpoint_dir / "config.json").write_text('{"model_type": "gpt2"}', encoding="utf-8")

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
    res = exporter.export("qwen_exp", quantization="Q4_K_M", run_conversion=False)

    assert res["status"] == "success"
    assert res["conversion_completed"] is False
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


def test_gguf_export_raises_when_conversion_requested_without_tools(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text('{"model_type": "gpt2"}', encoding="utf-8")

    mock_reg = MagicMock()
    mock_reg.get_experiment_record.return_value = {
        "id": "qwen_exp",
        "config": {"architecture": {"type": "llm", "llm_config": {"model_id": "gpt2"}}},
        "checkpoint_path": str(checkpoint_dir),
    }

    exporter = GGUFExporter(mock_reg)
    with patch.object(exporter, "resolve_tools", return_value=(None, None)):
        with pytest.raises(RuntimeError, match="missing required llama.cpp tool"):
            exporter.export("qwen_exp", quantization="Q4_K_M", run_conversion=True)


def test_gguf_real_conversion_pipeline_f16_and_quantized(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text('{"model_type": "gpt2"}', encoding="utf-8")

    # Create a real executable convert_hf_to_gguf.py stub that writes a valid GGUF v3 binary
    convert_script = tmp_path / "convert_hf_to_gguf.py"
    convert_script.write_text(
        "import sys, struct\n"
        "args = sys.argv[1:]\n"
        "out_idx = args.index('--outfile') + 1\n"
        "outfile = args[out_idx]\n"
        "with open(outfile, 'wb') as f:\n"
        "    f.write(struct.pack('<4sIQQ', b'GGUF', 3, 8, 4) + b'F16PAYLOAD')\n",
        encoding="utf-8",
    )

    # Create a real executable llama-quantize stub that verifies input F16 GGUF and writes quantized GGUF
    quantize_bin = tmp_path / "llama-quantize"
    quantize_bin.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, struct\n"
        "src, dst, qtype = sys.argv[1], sys.argv[2], sys.argv[3]\n"
        "with open(src, 'rb') as f:\n"
        "    hdr = f.read(4)\n"
        "assert hdr == b'GGUF'\n"
        "with open(dst, 'wb') as f:\n"
        "    f.write(struct.pack('<4sIQQ', b'GGUF', 3, 8, 5) + qtype.encode('utf-8'))\n",
        encoding="utf-8",
    )
    quantize_bin.chmod(quantize_bin.stat().st_mode | stat.S_IEXEC)

    mock_reg = MagicMock()
    mock_reg.get_experiment_record.return_value = {
        "id": "exp_conv",
        "config": {"architecture": {"type": "llm", "llm_config": {"model_id": "gpt2"}}},
        "checkpoint_path": str(checkpoint_dir),
    }

    # 1. F16 conversion: does NOT require llama-quantize
    exporter_f16 = GGUFExporter(
        mock_reg,
        convert_script_path=str(convert_script),
        llama_quantize_bin=None,
    )
    with patch("shutil.which", return_value=None):
        res_f16 = exporter_f16.export(
            "exp_conv",
            quantization="F16",
            output_dir=str(tmp_path / "out_f16"),
            run_conversion=True,
        )
    assert res_f16["conversion_completed"] is True
    assert os.path.isfile(res_f16["target_gguf_path"])
    assert res_f16["gguf_validation"]["valid"] is True
    assert res_f16["gguf_validation"]["tensor_count"] == 8

    # 2. Q4_K_M conversion: runs both convert_hf_to_gguf.py and llama-quantize
    exporter_q4 = GGUFExporter(
        mock_reg,
        convert_script_path=str(convert_script),
        llama_quantize_bin=str(quantize_bin),
    )
    res_q4 = exporter_q4.export(
        "exp_conv",
        quantization="Q4_K_M",
        output_dir=str(tmp_path / "out_q4"),
        run_conversion=True,
    )
    assert res_q4["conversion_completed"] is True
    assert os.path.isfile(res_q4["target_gguf_path"])
    assert res_q4["gguf_validation"]["valid"] is True
    assert res_q4["gguf_validation"]["metadata_kv_count"] == 5
