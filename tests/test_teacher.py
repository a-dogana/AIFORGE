import pytest
import torch
import torch.nn as nn
from aiforge.core.config import ExperimentConfig
from aiforge.training.teacher import TeacherEngine
from aiforge.training.trainer import TrainingEngine
from tests.test_training import DummyLoader, DummyModel
from unittest.mock import patch, MagicMock

import os
import shutil

@pytest.fixture
def temp_registry(tmp_path):
    from aiforge.experiments.registry import ExperimentRegistry
    reg_dir = tmp_path / "registry"
    os.makedirs(reg_dir)
    reg = ExperimentRegistry(str(reg_dir))
    yield reg
    shutil.rmtree(reg_dir)

def test_teacher_engine_loads_and_freezes(temp_registry):
    # Create a dummy experiment in the registry
    config_dict = {
        "experiment_name": "Teacher",
        "data": {"type": "digits_builtin"},
        "architecture": {"type": "mlp", "mlp_config": {"input_size": 10, "hidden_sizes": [8], "output_size": 2}},
        "batch_size": 4,
        "epochs": 1,
        "device": "cpu"
    }
    config = ExperimentConfig(**config_dict)
    
    # Save a dummy state
    from aiforge.models import ArchitectureEngine
    dummy_model = ArchitectureEngine.get_model(config)
    exp_id = temp_registry.save_experiment(config, {"best_model_state": dummy_model.state_dict()}, {}, "cpu")
    
    # Init TeacherEngine
    with patch("aiforge.training.teacher.ExperimentRegistry", return_value=temp_registry):
        teacher = TeacherEngine(exp_id, torch.device("cpu"))
        
    # Check it's frozen
    for param in teacher.model.parameters():
        assert not param.requires_grad
        
    # Check eval mode
    assert not teacher.model.training
    
    # Check output
    x = torch.randn(2, 10)
    logits = teacher.get_logits(x)
    assert logits.shape == (2, 2)
    assert not logits.requires_grad
    
def test_kd_loss_is_calculated():
    config_dict = {
        "experiment_name": "Student",
        "data": {"type": "digits_builtin"},
        "architecture": {"type": "mlp", "mlp_config": {"input_size": 10, "hidden_sizes": [4], "output_size": 2}},
        "batch_size": 4,
        "epochs": 1,
        "device": "cpu",
        "distillation": {
            "enabled": True,
            "teacher_experiment_id": "dummy",
            "temperature": 2.0,
            "alpha": 0.5
        }
    }
    config = ExperimentConfig(**config_dict)
    
    trainer = TrainingEngine(config)
    student = DummyModel()
    
    train_loader = DummyLoader(1, 4)
    val_loader = DummyLoader(1, 4)
    
    # Mock teacher engine
    mock_teacher = MagicMock()
    mock_teacher.get_logits.return_value = torch.randn(4, 2)
    
    res = trainer.train(student, train_loader, val_loader, teacher_engine=mock_teacher)
    
    assert "kd_loss" in res["history"]
    assert "supervised_loss" in res["history"]
    assert len(res["history"]["kd_loss"]) == 1
