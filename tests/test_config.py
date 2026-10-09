import pytest
import os
import yaml
from pydantic import ValidationError
from aiforge.core.config import ExperimentConfig

def test_valid_config():
    config_dict = {
        "experiment_name": "Test",
        "data": {
            "type": "digits_builtin"
        },
        "architecture": {
            "type": "mlp",
            "mlp_config": {
                "input_size": 64,
                "hidden_sizes": [128, 64],
                "output_size": 10
            }
        },
        "batch_size": 16,
        "epochs": 5
    }
    config = ExperimentConfig(**config_dict)
    assert config.data.type == "digits_builtin"
    assert config.batch_size == 16
    assert config.epochs == 5

def test_missing_dataset():
    with pytest.raises(ValidationError):
        # data is required
        ExperimentConfig(batch_size=16)

def test_yaml_parsing(tmp_path):
    yaml_content = {
        "experiment_name": "YAML Test",
        "data": {"type": "digits_builtin"},
        "architecture": {"type": "mlp", "mlp_config": {"input_size": 1, "hidden_sizes": [1], "output_size": 1}},
        "batch_size": 8
    }
    yaml_file = tmp_path / "test_config.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)
        
    config = ExperimentConfig.from_yaml(str(yaml_file))
    assert config.experiment_name == "YAML Test"
    assert config.batch_size == 8

def test_invalid_yaml_parsing(tmp_path):
    yaml_content = {
        "experiment_name": "Bad",
        "data": {"type": "csv"}, # missing csv_config
        "architecture": {"type": "cnn", "cnn_config": {"output_size": 2}}, # CNN not compatible with CSV
        "batch_size": 8
    }
    yaml_file = tmp_path / "bad_config.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)
        
    with pytest.raises(ValueError, match="Configuration validation failed"):
        ExperimentConfig.from_yaml(str(yaml_file))
