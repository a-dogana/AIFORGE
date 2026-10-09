import os
import json
import uuid
import datetime
import torch
from typing import Dict, Any, Tuple, Optional, List
from aiforge.core.config import ExperimentConfig

class ExperimentRegistry:
    def __init__(self, registry_dir: str = "registry"):
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)
        self.checkpoints_dir = os.path.join(registry_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
    def save_experiment(self, config: ExperimentConfig, training_results: Dict[str, Any], evaluation_results: Dict[str, Any], device_used: str, preprocessing_artifacts: Dict[str, Any] = None, experiment_id: Optional[str] = None) -> str:
        """
        Saves the experiment configuration, results, best model state, last training state, and preprocessing artifacts.
        Allows passing an existing experiment_id when resuming.
        """
        if preprocessing_artifacts is None:
            preprocessing_artifacts = {}
        if not experiment_id:
            experiment_id = str(uuid.uuid4())[:8]
            
        timestamp = datetime.datetime.now().isoformat()
        
        best_model_state = training_results.pop("best_model_state", None)
        last_training_state = training_results.pop("last_training_state", None)
        # Handle v0.2 compatibility where it was called 'model_state'
        if not best_model_state and "model_state" in training_results:
             best_model_state = training_results.pop("model_state")
             
        checkpoint_path = None
        training_state_path = None
        
        if best_model_state is not None:
            checkpoint_path = os.path.join(self.checkpoints_dir, f"model_best_{experiment_id}.pt")
            try:
                torch.save(best_model_state, checkpoint_path)
            except Exception as e:
                print(f"Error saving best model checkpoint: {e}")
                
        if last_training_state is not None:
            training_state_path = os.path.join(self.checkpoints_dir, f"training_state_{experiment_id}.pt")
            try:
                # Cannot use weights_only=True for full training state (contains optimizer, history dict, etc.)
                torch.save(last_training_state, training_state_path)
            except Exception as e:
                print(f"Error saving training state checkpoint: {e}")
        
        record = {
            "id": experiment_id,
            "experiment_name": getattr(config, "experiment_name", "N/A"),
            "timestamp": timestamp,
            "device_used": device_used,
            "checkpoint_path": checkpoint_path,
            "training_state_path": training_state_path,
            "schema_version": "0.5.0",
            "config": config.model_dump() if hasattr(config, "model_dump") else config.dict(),
            "preprocessing_artifacts": preprocessing_artifacts,
            "distillation_info": training_results.pop("distillation_info", {}),
            "results": {
                "training": training_results,
                "evaluation": evaluation_results
            }
        }
        
        # Save JSON record
        file_path = os.path.join(self.registry_dir, f"exp_{experiment_id}.json")
        with open(file_path, "w") as f:
            json.dump(record, f, indent=4)
            
        print(f"Experiment saved to registry: {file_path}")
        return experiment_id

    def load_experiment(self, experiment_id: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Loads an experiment record and its associated BEST model state_dict for inference.
        """
        file_path = os.path.join(self.registry_dir, f"exp_{experiment_id}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Experiment record not found for ID: {experiment_id}")
            
        with open(file_path, "r") as f:
            record = json.load(f)
            
        checkpoint_path = record.get("checkpoint_path")
        model_state = None
        
        # Backward compatibility for v0.2: if checkpoint_path doesn't exist, try the old naming scheme
        if not checkpoint_path or not os.path.exists(checkpoint_path):
             old_path = os.path.join(self.checkpoints_dir, f"model_{experiment_id}.pt")
             if os.path.exists(old_path):
                 checkpoint_path = old_path
                 
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                model_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except TypeError:
                model_state = torch.load(checkpoint_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"Best model checkpoint file not found at expected path: {checkpoint_path}")
                
        return record, model_state
        
    def load_training_state(self, experiment_id: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Loads the experiment record and the FULL training state for resuming.
        """
        file_path = os.path.join(self.registry_dir, f"exp_{experiment_id}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Experiment record not found for ID: {experiment_id}")
            
        with open(file_path, "r") as f:
            record = json.load(f)
            
        training_state_path = record.get("training_state_path")
        training_state = None
        
        if training_state_path and os.path.exists(training_state_path):
            # weights_only=False because this dict contains non-tensor objects (optimizer state, dicts)
            # This is safe assuming we trust the files we wrote locally.
            training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        else:
            raise FileNotFoundError(f"Training state checkpoint not found at: {training_state_path}")
            
        return record, training_state

    def list_experiments(self) -> List[Dict[str, Any]]:
        experiments = []
        for filename in os.listdir(self.registry_dir):
            if filename.startswith("exp_") and filename.endswith(".json"):
                with open(os.path.join(self.registry_dir, filename), "r") as f:
                    try:
                        record = json.load(f)
                        experiments.append(record)
                    except json.JSONDecodeError:
                        print(f"Warning: Corrupted JSON record {filename}")
        experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return experiments
