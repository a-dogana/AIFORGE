import pytest
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from aiforge.core.config import ExperimentConfig
from aiforge.data.dataset import DataEngine

@pytest.fixture
def dummy_csv(tmp_path):
    csv_file = tmp_path / "dummy.csv"
    data = {
        "feature1": [1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10],
        "feature2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "exclude": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return str(csv_file)

@pytest.fixture
def dummy_images(tmp_path):
    img_dir = tmp_path / "images"
    os.makedirs(img_dir / "class0")
    os.makedirs(img_dir / "class1")
    
    # Create 10 dummy images per class
    for i in range(10):
        img = Image.fromarray(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
        img.save(img_dir / "class0" / f"img_{i}.jpg")
        img.save(img_dir / "class1" / f"img_{i}.jpg")
        
    return str(img_dir)

def test_csv_loading(dummy_csv):
    config_dict = {
        "experiment_name": "CSV Test",
        "dataset_name": "dummy",
        "data": {
            "type": "csv",
            "csv_config": {
                "path": dummy_csv,
                "target_column": "target",
                "exclude_columns": ["exclude"],
                "missing_values": "mean"
            }
        },
        "architecture": {
            "type": "mlp",
            "mlp_config": {
                "input_size": 2,
                "hidden_sizes": [10],
                "output_size": 2
            }
        },
        "batch_size": 2,
        "epochs": 1
    }
    
    config = ExperimentConfig(**config_dict)
    train_loader, val_loader, test_loader, artifacts = DataEngine.get_dataloaders(config)
    
    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
    assert len(test_loader.dataset) > 0
    
    # Check artifacts
    assert "scaler" in artifacts
    assert "mean" in artifacts["scaler"]
    
    for inputs, targets in train_loader:
        assert inputs.shape[1] == 2 # 2 features
        break

def test_image_loading(dummy_images):
    config_dict = {
        "experiment_name": "Image Test",
        "dataset_name": "dummy",
        "data": {
            "type": "images",
            "image_config": {
                "path": dummy_images,
                "image_size": [32, 32],
                "mode": "single"
            }
        },
        "architecture": {
            "type": "cnn",
            "cnn_config": {
                "in_channels": 3,
                "conv_layers": [8],
                "hidden_sizes": [16],
                "output_size": 2
            }
        },
        "batch_size": 4,
        "epochs": 1
    }
    
    config = ExperimentConfig(**config_dict)
    train_loader, val_loader, test_loader, artifacts = DataEngine.get_dataloaders(config)
    
    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
    assert len(test_loader.dataset) > 0
    
    for inputs, targets in train_loader:
        assert inputs.shape == (4, 3, 32, 32)
        break
