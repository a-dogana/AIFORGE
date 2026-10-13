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
        
    def prepare(self):
        from aiforge.data.dataset import DataEngine
        from aiforge.models import ArchitectureEngine
        
        self.train_loader, self.val_loader, _, _ = DataEngine.get_dataloaders(self.config)
        self.model = ArchitectureEngine.get_model(self.config)

    def train(self) -> Dict[str, Any]:
        from aiforge.training.trainer import TrainingEngine
        trainer = TrainingEngine(self.config)
        self.results = trainer.train(self.model, self.train_loader, self.val_loader)
        return self.results

    def save(self, path: str):
        if self.model:
            torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))


class LLMAdapter(TrainingAdapterBase):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.engine = None
        
    def prepare(self):
        from aiforge.llm.engine import LLMEngine
        self.engine = LLMEngine(self.config)
        self.engine.load()

    def train(self) -> Dict[str, Any]:
        from aiforge.llm.engine import LLMEngine
        if self.config.data.type != "csv":
             raise ValueError("LLM training currently expects 'csv' or jsonl data type.")
        
        # Simplified: pass the path directly to engine
        dataset_path = self.config.data.csv_config.path if self.config.data.csv_config else None
        if not dataset_path:
             raise ValueError("LLM training requires dataset path in config.data.csv_config.path")

        # Fallback to backend logic
        if self.config.backend == "soup":
            from aiforge.llm.backends.soup import SoupBackend
            backend = SoupBackend(self.config)
            self.results = backend.train(dataset_path)
            # SoupBackend handles its own saving directly to registry within the run if integrated, but let's assume it returns standard format.
            # Actually, `engine.train(dataset)` is what `run_llm_train` calls.
        else:
            self.results = self.engine.train(dataset_path)
        
        return self.results

    def save(self, path: str):
        if self.engine and getattr(self.engine, 'model', None):
            self.engine.model.save_pretrained(path)
        if self.engine and getattr(self.engine, 'tokenizer', None):
            self.engine.tokenizer.save_pretrained(path)


class UnifiedOrchestrator:
    """
    Core orchestrator that selects the appropriate family adapter based on architecture.
    """
    def __init__(self, config: ExperimentConfig, registry: ExperimentRegistry):
        self.config = config
        self.registry = registry
        
        if config.architecture.type in ["mlp", "cnn"]:
            self.adapter = ClassicalAdapter(config)
        elif config.architecture.type == "llm":
            self.adapter = LLMAdapter(config)
        else:
            raise ValueError(f"UNSUPPORTED: Architecture '{config.architecture.type}' is not recognized.")

    def run(self):
        print(f"--- Unified Orchestrator: Starting {self.config.architecture.type.upper()} family training ---")
        self.adapter.prepare()
        results = self.adapter.train()
        
        # Save to registry
        import uuid
        import time
        exp_id = self.config.experiment_name.replace(" ", "_") + "_" + str(uuid.uuid4())[:8]
        reg_dir = getattr(self.registry, 'registry_dir', getattr(self.registry, 'registry_path', 'registry'))
        artifact_path = os.path.join(reg_dir, exp_id)
        os.makedirs(artifact_path, exist_ok=True)
        
        # Serialize config
        import yaml
        with open(os.path.join(artifact_path, "config.yaml"), "w") as f:
            yaml.dump(self.config.model_dump(), f)
            
        # Adapter save
        self.adapter.save(artifact_path)
        
        record = {
            "id": exp_id,
            "experiment_name": self.config.experiment_name,
            "config": self.config.model_dump(),
            "results": results,
            "timestamp": time.time(),
            "checkpoint_path": artifact_path
        }
        self.registry.save_experiment_record(exp_id, record)
        print(f"--- Unified Orchestrator: Finished. Saved as {exp_id} ---")
        return exp_id
