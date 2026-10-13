import pytest
import os
from aiforge.experiments.registry import ExperimentRegistry
from aiforge.deployment.ollama_export import OllamaExporter

def test_ollama_exporter(tmp_path):
    registry = ExperimentRegistry(str(tmp_path))
    ckpt_dir = tmp_path / "llm_ckpt"
    ckpt_dir.mkdir()

    config = {
        "experiment_name": "CoderSpecialist",
        "domain": "llm",
        "task": "generation",
        "data": {"type": "csv"},
        "architecture": {"type": "llm", "llm_config": {"model_id": "qwen"}}
    }
    registry.save_experiment_record("exp_coder", {
        "id": "exp_coder",
        "experiment_name": "CoderSpecialist",
        "config": config,
        "checkpoint_path": str(ckpt_dir)
    })

    exporter = OllamaExporter(registry)
    res = exporter.export("exp_coder", ollama_model_name="my-coder")

    assert res["status"] == "success"
    assert res["ollama_model_name"] == "my-coder"
    assert os.path.exists(res["modelfile_path"])
    with open(res["modelfile_path"]) as f:
        content = f.read()
    assert f"FROM {os.path.abspath(str(ckpt_dir))}" in content
    assert "PARAMETER temperature 0.7" in content

def test_ollama_exporter_rejects_non_llm(tmp_path):
    registry = ExperimentRegistry(str(tmp_path))
    config = {
        "experiment_name": "CNN_Vision",
        "architecture": {"type": "cnn"}
    }
    registry.save_experiment_record("exp_cnn", {
        "id": "exp_cnn",
        "config": config,
        "checkpoint_path": str(tmp_path)
    })
    exporter = OllamaExporter(registry)
    with pytest.raises(ValueError, match="Only LLM models"):
        exporter.export("exp_cnn")
