import torch
import torch.nn as nn
from typing import List

class BehaviorEncoder(nn.Module):
    """
    Multi-layer perceptron for encoding reviewer behavioral features.
    
    Processes handcrafted behavioral features (e.g., posting rate, rating deviation,
    account age) into a dense behavioral representation.
    """
    def __init__(self, input_dim: int = 15, hidden_dims: List[int] = [128, 128], dropout: float = 0.3):
        super(BehaviorEncoder, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        self.mlp = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the behavioral encoder.
        
        Args:
            features (torch.Tensor): Behavioral features tensor, shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Behavioral embedding, shape (batch_size, hidden_dims[-1]).
        """
        return self.mlp(features)
