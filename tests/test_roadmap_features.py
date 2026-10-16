import pytest
import os
import torch
import math
from unittest.mock import MagicMock

from aiforge.core.config import ExperimentConfig
from aiforge.experiments.registry import ExperimentRegistry
from aiforge.core.merge_lab import MergeLab, MergeRequest, slerp, dare_linear
from aiforge.evaluation.merge_eval import MergeEvaluator
from aiforge.deployment.ollama_export import OllamaExporter
from aiforge.core.resources import ResourceChecker
from aiforge.models import ArchitectureEngine

def test_slerp_and_dare_math():
    t0 = torch.tensor([1.0, 0.0, 0.0])
    t1 = torch.tensor([0.0, 1.0, 0.0])
    
    # SLERP midpoint
    mid = slerp(t0, t1, 0.5)
    assert abs(torch.norm(mid).item() - 1.0) < 1e-4
    assert torch.allclose(mid[0], mid[1])

    # DARE linear
    dare_res = dare_linear(t0, t1, alpha=0.5, drop_rate=0.0)
    assert torch.allclose(dare_res, 0.5 * t0 + 0.5 * t1)

def test_slerp_merge_lab_execution(tmp_path):
    registry = ExperimentRegistry(str(tmp_path))
    lab = MergeLab(registry)

    config = {
        "experiment_name": "MLP_Model",
        "domain": "tabular",
        "task": "classification",
        "data": {"type": "digits_builtin"},
        "architecture": {
            "type": "mlp",
            "mlp_config": {"input_size": 4, "hidden_sizes": [4], "output_size": 2}
        }
    }
    cfg_obj = ExperimentConfig(**config)
    m_a = ArchitectureEngine.get_model(cfg_obj)
    m_b = ArchitectureEngine.get_model(cfg_obj)

    dir_a = tmp_path / "exp_a"
    dir_b = tmp_path / "exp_b"
    dir_a.mkdir()
    dir_b.mkdir()

    torch.save(m_a.state_dict(), dir_a / "model.pt")
    torch.save(m_b.state_dict(), dir_b / "model.pt")

    registry.save_experiment_record("exp_a", {"id": "exp_a", "config": config, "checkpoint_path": str(dir_a)})
    registry.save_experiment_record("exp_b", {"id": "exp_b", "config": config, "checkpoint_path": str(dir_b)})

    # Test SLERP strategy
    req = MergeRequest(strategy="slerp", model_a_id="exp_a", model_b_id="exp_b", alpha=0.5, output_name="Merged_SLERP")
    val = lab.validate_pre_merge(req)
    assert val["status"] == "COMPATIBLE"

    out_id = lab.execute_merge(req)
    assert out_id is not None
    rec_out = registry.get_experiment_record(out_id)
    assert rec_out["checkpoint_path"].endswith(".pt")
    assert os.path.isfile(rec_out["checkpoint_path"])
    assert rec_out["results"]["merge_strategy"] == "slerp"

def test_resource_checker_nofile_guard():
    res = ResourceChecker.check_and_raise_nofile_limit(target_limit=1024)
    assert "supported" in res
    assert "soft" in res
    assert "hard" in res

def test_ollama_exporter_pipeline(tmp_path):
    registry = ExperimentRegistry(str(tmp_path))
    ckpt_dir = tmp_path / "ckpt_llm"
    ckpt_dir.mkdir()

    cfg = {
        "experiment_name": "Qwen_Specialized",
        "domain": "llm",
        "task": "generation",
        "architecture": {"type": "llm", "llm_config": {"model_id": "qwen0.5b"}}
    }
    registry.save_experiment_record("qwen_spec", {
        "id": "qwen_spec",
        "experiment_name": "Qwen_Specialized",
        "config": cfg,
        "checkpoint_path": str(ckpt_dir)
    })

    exporter = OllamaExporter(registry)
    export_res = exporter.export("qwen_spec")
    assert export_res["status"] == "success"
    assert os.path.exists(export_res["modelfile_path"])
    assert "ollama create" in export_res["create_command"]
