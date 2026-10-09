import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from aiforge.experiments.registry import ExperimentRegistry
from aiforge.core.config import ExperimentConfig
from aiforge.models import ArchitectureEngine

class TeacherEngine:
    """
    Loads and wraps a teacher model from the registry for knowledge distillation.
    Ensures the model is frozen and in evaluation mode.
    """
    def __init__(self, teacher_experiment_id: str, device: torch.device):
        self.teacher_experiment_id = teacher_experiment_id
        self.device = device
        self.registry = ExperimentRegistry()
        
        try:
            record, model_state = self.registry.load_experiment(teacher_experiment_id)
        except Exception as e:
            raise RuntimeError(f"Failed to load teacher experiment {teacher_experiment_id}: {e}")
            
        config_dict = record.get("config", {})
        try:
            self.teacher_config = ExperimentConfig(**config_dict)
        except Exception as e:
            raise RuntimeError(f"Teacher config is invalid: {e}")
            
        self.preprocessing_artifacts = record.get("preprocessing_artifacts", {})
        self.model = ArchitectureEngine.get_model(self.teacher_config)
        
        if model_state is None:
            raise RuntimeError(f"Teacher model state is missing for ID {teacher_experiment_id}.")
            
        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze teacher parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def validate_compatibility(self, student_config: ExperimentConfig):
        """
        Validates that the teacher and student configurations are compatible.
        """
        if self.teacher_config.task != student_config.task:
            raise ValueError("Teacher and student tasks do not match.")
            
        if self.teacher_config.data.type != student_config.data.type:
            raise ValueError("Teacher and student data types do not match.")
            
        # Ensure output sizes match
        teacher_out = self._get_output_size(self.teacher_config)
        student_out = self._get_output_size(student_config)
        if teacher_out != student_out:
            raise ValueError(f"Output sizes do not match: Teacher={teacher_out}, Student={student_out}")
            
    def _get_output_size(self, config: ExperimentConfig) -> int:
        if config.architecture.type == "mlp":
            return config.architecture.mlp_config.output_size
        elif config.architecture.type == "cnn":
            return config.architecture.cnn_config.output_size
        return -1
        
    def get_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Returns the teacher logits for a batch of inputs without tracking gradients.
        """
        with torch.no_grad():
            return self.model(inputs)
            
    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
