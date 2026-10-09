import pytest
import torch
import torch.nn as nn
from aiforge.core.config import ExperimentConfig
from aiforge.training.trainer import TrainingEngine

# Dummy Dataset and Loader
class DummyDataset:
    def __init__(self, size):
        self.size = size
    def __len__(self):
        return self.size

class DummyLoader:
    def __init__(self, num_batches, batch_size):
        self.dataset = DummyDataset(num_batches * batch_size)
        self.num_batches = num_batches
        self.batch_size = batch_size
        self._inputs = torch.randn(batch_size, 10)
        self._targets = torch.zeros(batch_size, dtype=torch.long)
        
    def __iter__(self):
        for _ in range(self.num_batches):
            yield self._inputs, self._targets
            
    def __len__(self):
        return self.num_batches

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

def test_training_best_checkpoint_min_mode():
    config = ExperimentConfig(
        data={"type": "digits_builtin"},
        architecture={"type": "mlp", "mlp_config": {"input_size": 1, "hidden_sizes": [1], "output_size": 1}},
        epochs=3,
        device="cpu"
    )
    # By default, early stopping is disabled, but best checkpoint is still tracked
    # It tracks val_loss (min) by default if we don't enable it or just tracks best metric val.
    
    trainer = TrainingEngine(config)
    model = DummyModel()
    train_loader = DummyLoader(2, 4)
    val_loader = DummyLoader(1, 4)
    
    res = trainer.train(model, train_loader, val_loader)
    
    assert "best_epoch" in res
    assert "best_model_state" in res
    assert "last_training_state" in res

def test_early_stopping_max_mode():
    config = ExperimentConfig(
        data={"type": "digits_builtin"},
        architecture={"type": "mlp", "mlp_config": {"input_size": 1, "hidden_sizes": [1], "output_size": 1}},
        epochs=10,
        device="cpu"
    )
    config.early_stopping.enabled = True
    config.early_stopping.metric = "val_accuracy"
    config.early_stopping.mode = "max"
    config.early_stopping.patience = 2
    config.early_stopping.min_epochs = 1
    
    trainer = TrainingEngine(config)
    model = DummyModel()
    train_loader = DummyLoader(2, 4)
    val_loader = DummyLoader(1, 4)
    
    res = trainer.train(model, train_loader, val_loader)
    
    assert res["total_epochs_run"] < 10
    assert "Early stopping triggered" in res["stop_reason"]

def test_resume_training():
    config = ExperimentConfig(
        data={"type": "digits_builtin"},
        architecture={"type": "mlp", "mlp_config": {"input_size": 1, "hidden_sizes": [1], "output_size": 1}},
        epochs=4,
        device="cpu"
    )
    trainer = TrainingEngine(config)
    model = DummyModel()
    train_loader = DummyLoader(2, 4)
    val_loader = DummyLoader(1, 4)
    
    # First run: train for 2 epochs by mocking the config temporarily
    config.epochs = 2
    res1 = trainer.train(model, train_loader, val_loader)
    
    # Second run: resume from last state and finish up to 4 epochs
    config.epochs = 4
    resume_state = res1["last_training_state"]
    res2 = trainer.train(model, train_loader, val_loader, resume_state=resume_state)
    
    assert res2["total_epochs_run"] == 2 # 2 more epochs (epoch 3 and 4)
    assert len(res2["history"]["train_loss"]) == 4 # Total history should be 4
