import pytest
import os
import shutil
import torch
from aiforge.experiments.registry import ExperimentRegistry
from aiforge.core.config import ExperimentConfig

@pytest.fixture
def temp_registry(tmp_path):
    registry_dir = tmp_path / "registry"
    registry = ExperimentRegistry(registry_dir=str(registry_dir))
    yield registry
    shutil.rmtree(registry_dir)

def test_save_and_load_experiment(temp_registry):
    config = ExperimentConfig(
        experiment_name="Registry Test",
        data={"type": "digits_builtin"},
        architecture={"type": "mlp", "mlp_config": {"input_size": 1, "hidden_sizes": [1], "output_size": 1}}
    )
    
    dummy_model_state = {"layer.weight": torch.tensor([1.0, 2.0])}
    training_results = {"train_loss": [0.1], "model_state": dummy_model_state}
    eval_results = {"accuracy": 0.95}
    
    exp_id = temp_registry.save_experiment(config, training_results, eval_results, "cpu")
    
    # Load
    record, loaded_state = temp_registry.load_experiment(exp_id)
    
    assert record["id"] == exp_id
    assert record["experiment_name"] == "Registry Test"
    assert record["results"]["evaluation"]["accuracy"] == 0.95
    assert "model_state" not in record["results"]["training"]
    
    assert loaded_state is not None
    assert "layer.weight" in loaded_state
    assert torch.equal(loaded_state["layer.weight"], dummy_model_state["layer.weight"])

def test_load_missing_experiment(temp_registry):
    with pytest.raises(FileNotFoundError):
        temp_registry.load_experiment("invalid_id")

def test_missing_checkpoint(temp_registry):
    config = ExperimentConfig(
        experiment_name="Registry Test",
        data={"type": "digits_builtin"},
        architecture={"type": "mlp", "mlp_config": {"input_size": 1, "hidden_sizes": [1], "output_size": 1}}
    )
    exp_id = temp_registry.save_experiment(config, {"model_state": {"a": torch.tensor(1)}}, {}, "cpu")
    
    # Corrupt by deleting the checkpoint
    record, _ = temp_registry.load_experiment(exp_id)
    os.remove(record["checkpoint_path"])
    
    with pytest.raises(FileNotFoundError, match="Best model checkpoint file not found"):
        temp_registry.load_experiment(exp_id)

def test_registry_backward_compatibility(temp_registry):
    import json
    import uuid
    exp_id = str(uuid.uuid4())[:8]
    
    # Create a mock v0.2 JSON record and model file
    file_path = os.path.join(temp_registry.registry_dir, f"exp_{exp_id}.json")
    checkpoint_path = os.path.join(temp_registry.checkpoints_dir, f"model_{exp_id}.pt")
    
    torch.save({"dummy": torch.tensor([1, 2])}, checkpoint_path)
    
    record = {
        "id": exp_id,
        "checkpoint_path": checkpoint_path,
        "schema_version": "0.2.0",
        "results": {"training": {}, "evaluation": {}}
    }
    with open(file_path, "w") as f:
        json.dump(record, f)
        
    # Test that load_experiment can load it properly
    loaded_record, loaded_state = temp_registry.load_experiment(exp_id)
    
    assert loaded_record["id"] == exp_id
    assert loaded_state is not None
    assert "dummy" in loaded_state
