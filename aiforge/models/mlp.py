import torch
import torch.nn as nn
from aiforge.core.config import MLPConfig

class SimpleMLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super(SimpleMLP, self).__init__()
        
        layers = []
        input_dim = config.input_size
        
        for hidden_dim in config.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, config.output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
