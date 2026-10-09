import pytest
from unittest.mock import patch, MagicMock
from aiforge.core.config import ExperimentConfig
from aiforge.tuning.autotuner import AutoTuningEngine

def test_autotuning_disabled():
    config_dict = {
        "experiment_name": "Tuning Test",
        "data": {"type": "digits_builtin"},
        "architecture": {"type": "mlp", "mlp_config": {"input_size": 1, "hidden_sizes": [1], "output_size": 1}},
        "batch_size": 4,
        "epochs": 1,
        "tuning": {"enabled": False}
    }
    config = ExperimentConfig(**config_dict)
    with pytest.raises(ValueError, match="AutoTuning is not enabled"):
        AutoTuningEngine(config)

def test_apply_parameter():
    config_dict = {
        "experiment_name": "Tuning Test",
        "data": {"type": "digits_builtin"},
        "architecture": {"type": "mlp", "mlp_config": {"input_size": 1, "hidden_sizes": [1], "output_size": 1}},
        "batch_size": 4,
        "epochs": 1,
        "tuning": {"enabled": True, "trials": 1}
    }
    config = ExperimentConfig(**config_dict)
    tuner = AutoTuningEngine(config)
    
    cdict = config.model_dump()
    tuner._apply_parameter(cdict, "learning_rate", 0.05)
    tuner._apply_parameter(cdict, "distillation.temperature", 4.0)
    
    assert cdict["learning_rate"] == 0.05
    assert cdict["distillation"]["temperature"] == 4.0

def test_tuning_run():
    config_dict = {
        "experiment_name": "Tuning Test",
        "data": {"type": "digits_builtin"},
        "architecture": {"type": "mlp", "mlp_config": {"input_size": 1, "hidden_sizes": [1], "output_size": 1}},
        "batch_size": 4,
        "epochs": 1,
        "tuning": {
            "enabled": True, 
            "trials": 3,
            "metric": "val_loss",
            "mode": "min",
            "search_space": [
                {"name": "learning_rate", "type": "float", "low": 0.001, "high": 0.1, "log": True},
                {"name": "epochs", "type": "int", "low": 1, "high": 3}
            ]
        }
    }
    config = ExperimentConfig(**config_dict)
    tuner = AutoTuningEngine(config)
    
    def dummy_objective(cfg: ExperimentConfig):
        # We try to minimize this dummy metric
        # The smaller the learning rate and epochs, the better (just a dummy function)
        return cfg.learning_rate + cfg.epochs
        
    study = tuner.run(dummy_objective)
    assert len(study.trials) == 3
    
    best_config = tuner.get_best_config()
    assert best_config.learning_rate >= 0.001 and best_config.learning_rate <= 0.1
    assert best_config.epochs in [1, 2, 3]
    assert not best_config.tuning.enabled # Tuning should be disabled on the returned config
