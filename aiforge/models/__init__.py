import torch.nn as nn
from aiforge.core.config import ExperimentConfig
from aiforge.models.mlp import SimpleMLP
from aiforge.models.cnn import SimpleCNN

class ArchitectureEngine:
    @staticmethod
    def get_model(config: ExperimentConfig) -> nn.Module:
        arch_type = config.architecture.type
        
        if arch_type == "mlp":
            return SimpleMLP(config.architecture.mlp_config)
        elif arch_type == "cnn":
            return SimpleCNN(config.architecture.cnn_config)
        else:
            raise ValueError(f"Unsupported architecture type: {arch_type}")
