import os
import re
import json
import uuid
import datetime
import torch
from typing import Dict, Any, Tuple, Optional, List
from aiforge.core.config import ExperimentConfig

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class ExperimentRegistry:
    def __init__(self, registry_dir: str = "registry"):
        self.registry_dir = os.path.abspath(registry_dir)
        self.registry_path = self.registry_dir
        if os.path.lexists(self.registry_dir) and os.path.islink(self.registry_dir):
            raise ValueError(f"Security: Registry directory cannot be a symlink: '{registry_dir}'")
        os.makedirs(self.registry_dir, exist_ok=True)
        self.checkpoints_dir = self.resolve_safe_path("checkpoints", allow_create_dirs=True)

    def _validate_id(self, experiment_id: str) -> str:
        if not experiment_id or not isinstance(experiment_id, str):
            raise ValueError("Experiment ID must be a non-empty string.")
        if (
            ".." in experiment_id
            or "/" in experiment_id
            or "\\" in experiment_id
            or ":" in experiment_id
            or "\x00" in experiment_id
            or os.path.isabs(experiment_id)
            or _WIN_DRIVE_RE.match(experiment_id)
            or not _SAFE_ID_RE.match(experiment_id)
        ):
            raise ValueError(f"Security: Invalid or unsafe experiment ID: '{experiment_id}'")
        return experiment_id

    def validate_display_name(self, name: str) -> str:
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError("Experiment display name must be a non-empty string.")
        if (
            ".." in name
            or "/" in name
            or "\\" in name
            or "\x00" in name
            or os.path.isabs(name)
            or _WIN_DRIVE_RE.match(name.strip())
        ):
            raise ValueError(f"Security: Unsafe path characters in experiment name: '{name}'")
        return name

    def generate_experiment_id(self, display_name: Optional[str] = None) -> str:
        """Generate a safe, opaque experiment ID decoupled from the display name."""
        if display_name is not None:
            self.validate_display_name(display_name)
        return uuid.uuid4().hex[:8]

    def resolve_safe_path(self, *parts: str, allow_create_dirs: bool = False) -> str:
        """Resolve a path inside ``self.registry_dir`` and reject traversal, absolute paths, or symlinks."""
        reg_abs = os.path.abspath(self.registry_dir)
        if os.path.lexists(reg_abs) and os.path.islink(reg_abs):
            raise ValueError(f"Security: Registry root is a symlink: '{reg_abs}'")
        reg_real = os.path.realpath(reg_abs)

        cur = reg_abs
        for part in parts:
            self._validate_id(part)
            cur = os.path.join(cur, part)
            if os.path.lexists(cur):
                if os.path.islink(cur):
                    raise ValueError(f"Security: Symlink detected inside registry: '{cur}'")
                cur_real = os.path.realpath(cur)
                if os.path.commonpath([reg_real, cur_real]) != reg_real:
                    raise ValueError(f"Security: Path escapes registry directory: '{cur}'")

        parent_real = os.path.realpath(os.path.dirname(cur))
        if os.path.commonpath([reg_real, parent_real]) != reg_real:
            raise ValueError(f"Security: Parent path escapes registry directory: '{cur}'")

        if allow_create_dirs:
            os.makedirs(cur, exist_ok=True)
            if os.path.islink(cur) or os.path.commonpath([reg_real, os.path.realpath(cur)]) != reg_real:
                raise ValueError(f"Security: Created directory escapes registry: '{cur}'")

        return cur

    def save_experiment(self, config: ExperimentConfig, training_results: Dict[str, Any], evaluation_results: Dict[str, Any], device_used: str, preprocessing_artifacts: Dict[str, Any] = None, experiment_id: Optional[str] = None) -> str:
        """
        Saves the experiment configuration, results, best model state, last training state, and preprocessing artifacts.
        Allows passing an existing experiment_id when resuming.
        """
        if preprocessing_artifacts is None:
            preprocessing_artifacts = {}
        display_name = getattr(config, "experiment_name", "N/A")
        if display_name and display_name != "N/A":
            self.validate_display_name(display_name)
        if not experiment_id:
            experiment_id = self.generate_experiment_id(display_name if display_name != "N/A" else None)
        else:
            experiment_id = self._validate_id(experiment_id)

        # Pre-validate target record path before writing any checkpoint files
        file_path = self.resolve_safe_path(f"exp_{experiment_id}.json")
        timestamp = datetime.datetime.now().isoformat()
        
        best_model_state = training_results.pop("best_model_state", None)
        last_training_state = training_results.pop("last_training_state", None)
        # Handle v0.2 compatibility where it was called 'model_state'
        if not best_model_state and "model_state" in training_results:
             best_model_state = training_results.pop("model_state")
             
        checkpoint_path = training_results.pop("checkpoint_path", None)
        training_state_path = None
        
        if best_model_state is not None:
            checkpoint_path = self.resolve_safe_path("checkpoints", f"model_best_{experiment_id}.pt")
            try:
                torch.save(best_model_state, checkpoint_path)
            except Exception as e:
                print(f"Error saving best model checkpoint: {e}")
                
        if last_training_state is not None:
            training_state_path = self.resolve_safe_path("checkpoints", f"training_state_{experiment_id}.pt")
            try:
                # Cannot use weights_only=True for full training state (contains optimizer, history dict, etc.)
                torch.save(last_training_state, training_state_path)
            except Exception as e:
                print(f"Error saving training state checkpoint: {e}")
        
        record = {
            "id": experiment_id,
            "experiment_name": display_name,
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
        
        with open(file_path, "w") as f:
            json.dump(record, f, indent=4)
            
        print(f"Experiment saved to registry: {file_path}")
        return experiment_id

    def load_experiment(self, experiment_id: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Loads an experiment record and its associated BEST model state_dict for inference.
        """
        experiment_id = self._validate_id(experiment_id)
        file_path = self.resolve_safe_path(f"exp_{experiment_id}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Experiment record not found for ID: {experiment_id}")
            
        with open(file_path, "r") as f:
            record = json.load(f)
            
        checkpoint_path = record.get("checkpoint_path")
        model_state = None
        
        # Backward compatibility for v0.2: if checkpoint_path doesn't exist, try the old naming scheme
        if not checkpoint_path or not os.path.exists(checkpoint_path):
             old_path = self.resolve_safe_path("checkpoints", f"model_{experiment_id}.pt")
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

    def save_experiment_record(self, experiment_id: str, record: Dict[str, Any]):
        experiment_id = self._validate_id(experiment_id)
        file_path = self.resolve_safe_path(f"exp_{experiment_id}.json")
        with open(file_path, "w") as f:
            json.dump(record, f, indent=4)

    def get_experiment_record(self, experiment_id: str) -> Dict[str, Any]:
        """Loads only the JSON record without the weights."""
        experiment_id = self._validate_id(experiment_id)
        file_path = self.resolve_safe_path(f"exp_{experiment_id}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Experiment record not found for ID: {experiment_id}")
            
        with open(file_path, "r") as f:
            return json.load(f)
        
    def load_training_state(self, experiment_id: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Loads the experiment record and the FULL training state for resuming.
        """
        experiment_id = self._validate_id(experiment_id)
        file_path = self.resolve_safe_path(f"exp_{experiment_id}.json")
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
