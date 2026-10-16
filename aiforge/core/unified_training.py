import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from aiforge.core.config import ExperimentConfig
from aiforge.experiments.registry import ExperimentRegistry

class TrainingAdapterBase(ABC):
    def __init__(self, config: ExperimentConfig):
        self.config = config

    @abstractmethod
    def prepare(self):
        """Prepare datasets and models"""
        pass

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Run training logic"""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save trained artifact to registry path"""
        pass


class ClassicalAdapter(TrainingAdapterBase):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.preprocessing_artifacts = {}
        
    def prepare(self):
        from aiforge.data.dataset import DataEngine
        from aiforge.models import ArchitectureEngine
        
        self.train_loader, self.val_loader, self.test_loader, self.preprocessing_artifacts = DataEngine.get_dataloaders(self.config)
        self.model = ArchitectureEngine.get_model(self.config)

    def train(self) -> Dict[str, Any]:
        from aiforge.training.trainer import TrainingEngine
        trainer = TrainingEngine(self.config)
        self.results = trainer.train(self.model, self.train_loader, self.val_loader)
        return self.results

    def save(self, path: str) -> str:
        os.makedirs(path, exist_ok=True)
        pt_file = os.path.join(path, "model.pt")
        if self.model is not None:
            best_state = getattr(self, "results", {}).get("best_model_state") if hasattr(self, "results") and isinstance(self.results, dict) else None
            state_to_save = best_state if best_state is not None else self.model.state_dict()
            torch.save(state_to_save, pt_file)
        return pt_file


class LLMAdapter(TrainingAdapterBase):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.engine = None
        self.results: Dict[str, Any] = {}
        
    def prepare(self):
        if self.config.backend == "soup":
            return
        from aiforge.llm.engine import LLMEngine
        self.engine = LLMEngine(self.config)
        self.engine.load()

    def train(self) -> Dict[str, Any]:
        if self.config.data.type not in ("csv", "jsonl"):
            raise ValueError("LLM training expects 'csv' or 'jsonl' data type.")
        
        dataset_path = self.config.data.csv_config.path if self.config.data.csv_config else None
        if not dataset_path:
            raise ValueError("LLM training requires dataset path in config.data.csv_config.path")

        if self.config.backend == "soup":
            from aiforge.llm.backends.soup import SoupBackend
            backend = SoupBackend(self.config)
            self.results = backend.train(dataset_path)
        else:
            if self.engine is None:
                from aiforge.llm.engine import LLMEngine
                self.engine = LLMEngine(self.config)
                self.engine.load()
            self.results = self.engine.train(dataset_path)
        
        return self.results

    def save(self, path: str) -> str:
        import shutil
        os.makedirs(path, exist_ok=True)
        if self.config.backend == "soup":
            soup_ckpt = (self.results or {}).get("checkpoint_path")
            if soup_ckpt and os.path.isdir(soup_ckpt) and os.path.abspath(soup_ckpt) != os.path.abspath(path):
                shutil.copytree(soup_ckpt, path, dirs_exist_ok=True)
            return path

        llm_cfg = getattr(self.config.architecture, "llm_config", None)
        is_adapter = bool(llm_cfg and (getattr(llm_cfg, "lora_enabled", False) or getattr(llm_cfg, "qlora", False)))
        if self.engine is not None and is_adapter and hasattr(self.engine, "save_adapters"):
            self.engine.save_adapters(path)
            return path

        if self.engine and getattr(self.engine, "model", None) and hasattr(self.engine.model, "save_pretrained"):
            self.engine.model.save_pretrained(path)
        if self.engine and getattr(self.engine, "tokenizer", None) and hasattr(self.engine.tokenizer, "save_pretrained"):
            self.engine.tokenizer.save_pretrained(path)
        return path


class UnifiedOrchestrator:
    """
    Core orchestrator that selects the appropriate family adapter based on architecture.
    """
    def __init__(self, config: ExperimentConfig, registry: ExperimentRegistry):
        self.config = config
        self.registry = registry
        self.registry.validate_display_name(self.config.experiment_name)
        
        if config.architecture.type in ["mlp", "cnn"]:
            self.adapter = ClassicalAdapter(config)
        elif config.architecture.type == "llm":
            self.adapter = LLMAdapter(config)
        else:
            raise ValueError(f"UNSUPPORTED: Architecture '{config.architecture.type}' is not recognized.")

    def run(self, experiment_id: Optional[str] = None):
        print(f"--- Unified Orchestrator: Starting {self.config.architecture.type.upper()} family training ---")
        # Validate and pre-resolve safe ID & paths BEFORE training or creating any directory
        self.registry.validate_display_name(self.config.experiment_name)
        exp_id = (
            self.registry._validate_id(experiment_id)
            if experiment_id is not None
            else self.registry.generate_experiment_id(self.config.experiment_name)
        )
        self.registry.resolve_safe_path(exp_id)
        self.registry.resolve_safe_path(f"exp_{exp_id}.json")

        self.adapter.prepare()
        results = self.adapter.train()
        
        # Save to registry inside validated directory
        import yaml
        artifact_path = self.registry.resolve_safe_path(exp_id, allow_create_dirs=True)
        config_file = self.registry.resolve_safe_path(exp_id, "config.yaml")
        with open(config_file, "w") as f:
            yaml.dump(self.config.model_dump(), f)
            
        # Adapter save returns unified checkpoint_path (.pt file for mlp/cnn, directory for llm)
        saved_ckpt_path = self.adapter.save(artifact_path) or artifact_path
        clean_results = dict(results or {})
        clean_results.pop("best_model_state", None)
        clean_results.pop("last_training_state", None)
        clean_results.pop("checkpoint_path", None)
        
        record = {
            "id": exp_id,
            "experiment_name": self.config.experiment_name,
            "config": self.config.model_dump(),
            "results": clean_results,
            "preprocessing_artifacts": getattr(self.adapter, "preprocessing_artifacts", {}),
            "timestamp": self.registry.now_utc_iso(),
            "checkpoint_path": saved_ckpt_path
        }
        self.registry.save_experiment_record(exp_id, record)
        print(f"--- Unified Orchestrator: Finished. Saved as {exp_id} ---")
        return exp_id
