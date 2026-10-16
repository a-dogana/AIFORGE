import pytest
import os
import torch
from aiforge.core.config import ExperimentConfig
from aiforge.experiments.registry import ExperimentRegistry
from aiforge.core.merge_lab import MergeLab, MergeRequest

@pytest.fixture
def registry(tmp_path):
    return ExperimentRegistry(str(tmp_path))

def test_full_weight_merge_mlp(registry, tmp_path):
    lab = MergeLab(registry)
    
    # Create two fake MLP experiments with actual random weights saved
    config = {
        "experiment_name": "MLP_A",
        "domain": "tabular",
        "task": "classification",
        "data": {"type": "digits_builtin"},
        "architecture": {
            "type": "mlp",
            "mlp_config": {"input_size": 10, "hidden_sizes": [5], "output_size": 2}
        }
    }
    
    # Init actual models
    from aiforge.models import ArchitectureEngine
    cfg_obj = ExperimentConfig(**config)
    model_a = ArchitectureEngine.get_model(cfg_obj)
    model_b = ArchitectureEngine.get_model(cfg_obj)
    
    # Save them
    path_a = os.path.join(tmp_path, "A")
    path_b = os.path.join(tmp_path, "B")
    os.makedirs(path_a)
    os.makedirs(path_b)
    
    torch.save(model_a.state_dict(), os.path.join(path_a, "model.pt"))
    torch.save(model_b.state_dict(), os.path.join(path_b, "model.pt"))
    
    registry.save_experiment_record("A", {"id": "A", "config": config, "checkpoint_path": path_a})
    registry.save_experiment_record("B", {"id": "B", "config": config, "checkpoint_path": path_b})
    
    # 1. Test Compatibility
    req = MergeRequest(strategy="linear_weight", model_a_id="A", model_b_id="B", alpha=0.5, output_name="C")
    val = lab.validate_pre_merge(req)
    assert val["status"] == "COMPATIBLE"
    
    # 2. Execute Merge
    out_id = lab.execute_merge(req)
    
    # 3. Verify resulting weights (contract: MLP checkpoint_path is a .pt file, even when inputs A/B used legacy dir paths)
    rec_c = registry.get_experiment_record(out_id)
    assert rec_c is not None
    assert rec_c["checkpoint_path"].endswith(".pt")
    assert os.path.isfile(rec_c["checkpoint_path"])
    
    model_c = ArchitectureEngine.get_model(cfg_obj)
    model_c.load_state_dict(registry.load_classical_state_dict(rec_c))
    
    # Verify mathematically (w_c = 0.5 * w_a + 0.5 * w_b)
    for p_a, p_b, p_c in zip(model_a.parameters(), model_b.parameters(), model_c.parameters()):
        expected = 0.5 * p_a + 0.5 * p_b
        assert torch.allclose(expected, p_c)

def test_lora_merge_validation_mismatch(registry):
    lab = MergeLab(registry)
    
    # Try merging MLP with LoRA (Should fail)
    config = {
        "experiment_name": "MLP",
        "domain": "tabular",
        "task": "classification",
        "data": {"type": "digits_builtin"},
        "architecture": {
            "type": "mlp",
            "mlp_config": {"input_size": 10, "hidden_sizes": [5], "output_size": 2}
        }
    }
    registry.save_experiment_record("A", {"id": "A", "config": config, "checkpoint_path": ""})
    registry.save_experiment_record("B", {"id": "B", "config": config, "checkpoint_path": ""})
    
    req = MergeRequest(strategy="lora_merge", model_a_id="A", model_b_id="B", alpha=0.5, output_name="C")
    val = lab.validate_pre_merge(req)
    assert val["status"] == "INCOMPATIBLE"
    assert "LoRA merge currently only supported for 'llm'" in val["incompatibilities"][0]
