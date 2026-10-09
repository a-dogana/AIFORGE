import pytest
import torch
from aiforge.core.config import ExperimentConfig
from aiforge.models import ArchitectureEngine

def test_cnn_architecture():
    config_dict = {
        "experiment_name": "CNN Test",
        "dataset_name": "dummy",
        "data": {
            "type": "images",
            "image_config": {
                "path": "dummy",
                "image_size": [32, 32],
                "mode": "single"
            }
        },
        "architecture": {
            "type": "cnn",
            "cnn_config": {
                "in_channels": 3,
                "conv_layers": [8, 16],
                "hidden_sizes": [32],
                "output_size": 2
            }
        },
        "batch_size": 4,
        "epochs": 1
    }
    config = ExperimentConfig(**config_dict)
    
    model = ArchitectureEngine.get_model(config)
    
    # Dummy input: (batch_size, channels, height, width)
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    
    assert output.shape == (4, 2)

def test_incompatible_architecture():
    config_dict = {
        "experiment_name": "CNN Test",
        "dataset_name": "dummy",
        "data": {
            "type": "csv",
            "csv_config": {
                "path": "dummy",
                "target_column": "t"
            }
        },
        "architecture": {
            "type": "cnn",
            "cnn_config": {
                "in_channels": 3,
                "conv_layers": [8, 16],
                "hidden_sizes": [32],
                "output_size": 2
            }
        },
        "batch_size": 4,
        "epochs": 1
    }
    
    with pytest.raises(ValueError, match="CNN architecture is only compatible with 'images'"):
        ExperimentConfig(**config_dict)
