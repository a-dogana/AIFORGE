import copy
import optuna
import optuna.exceptions
from typing import Dict, Any, Callable
from aiforge.core.config import ExperimentConfig

class AutoTuningEngine:
    """
    Manages hyperparameter optimization using Optuna.
    """
    def __init__(self, base_config: ExperimentConfig):
        self.base_config = base_config
        self.tuning_config = base_config.tuning
        
        if not self.tuning_config.enabled:
            raise ValueError("AutoTuning is not enabled in the configuration.")
            
        self.study = optuna.create_study(
            study_name=self.tuning_config.study_name,
            storage=self.tuning_config.storage,
            direction="minimize" if self.tuning_config.mode == "min" else "maximize",
            load_if_exists=True
        )

    def _suggest_parameter(self, trial: optuna.Trial, param_space) -> Any:
        if param_space.type == "float":
            return trial.suggest_float(param_space.name, param_space.low, param_space.high, log=param_space.log)
        elif param_space.type == "int":
            return trial.suggest_int(param_space.name, int(param_space.low), int(param_space.high), log=param_space.log)
        elif param_space.type == "categorical":
            return trial.suggest_categorical(param_space.name, param_space.choices)
        else:
            raise ValueError(f"Unknown parameter type {param_space.type}")

    def _apply_parameter(self, config_dict: Dict, param_name: str, value: Any):
        """
        Applies a parameter to a nested config dictionary using dot notation (e.g., 'distillation.temperature')
        """
        keys = param_name.split('.')
        current = config_dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def run(self, objective_fn: Callable[[ExperimentConfig], float]) -> optuna.Study:
        """
        Runs the optimization loop.
        objective_fn should take an ExperimentConfig and return a float metric.
        """
        def optuna_objective(trial: optuna.Trial) -> float:
            config_dict = self.base_config.model_dump()
            
            # Sample and apply hyperparameters
            for param_space in self.tuning_config.search_space:
                val = self._suggest_parameter(trial, param_space)
                self._apply_parameter(config_dict, param_space.name, val)
                
            # Create a new config for this trial
            # Turn off tuning for the inner run to prevent recursive loops
            config_dict["tuning"]["enabled"] = False
            
            try:
                trial_config = ExperimentConfig(**config_dict)
            except Exception as e:
                # If the proposed configuration is invalid, prune the trial
                raise optuna.exceptions.TrialPruned(f"Invalid configuration: {e}")
                
            # Execute the objective
            return objective_fn(trial_config)

        self.study.optimize(optuna_objective, n_trials=self.tuning_config.trials)
        return self.study

    def get_best_config(self) -> ExperimentConfig:
        if len(self.study.trials) == 0 or self.study.best_trial is None:
            raise ValueError("No trials completed successfully.")
            
        best_params = self.study.best_trial.params
        config_dict = self.base_config.model_dump()
        
        for name, value in best_params.items():
            self._apply_parameter(config_dict, name, value)
            
        config_dict["tuning"]["enabled"] = False
        return ExperimentConfig(**config_dict)
