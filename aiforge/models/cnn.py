import torch
import torch.nn as nn
from aiforge.core.config import CNNConfig

class SimpleCNN(nn.Module):
    def __init__(self, config: CNNConfig):
        super(SimpleCNN, self).__init__()
        
        layers = []
        in_channels = config.in_channels
        
        for out_channels in config.conv_layers:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            
        self.features = nn.Sequential(*layers)
        
        # Adaptive pooling to ensure fixed output size before dense layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        dense_layers = []
        # Input to first dense layer will be (last out_channels) * 4 * 4
        in_features = config.conv_layers[-1] * 16 if config.conv_layers else in_channels * 16
        
        for hidden_dim in config.hidden_sizes:
            dense_layers.append(nn.Linear(in_features, hidden_dim))
            dense_layers.append(nn.ReLU())
            in_features = hidden_dim
            
        dense_layers.append(nn.Linear(in_features, config.output_size))
        
        self.classifier = nn.Sequential(*dense_layers)
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
