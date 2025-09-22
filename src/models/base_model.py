import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

class BaseModel(nn.Module):
    """Base class for all models in the project"""
    
    def __init__(self, 
                 input_dim: int,
                 num_classes: int,
                 hidden_dims: list = [128, 64],
                 dropout_rate: float = 0.2):
        """
        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(BaseModel, self).__init__()
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        # Add final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'input_dim': self.network[0].in_features,
            'num_classes': self.network[-1].out_features,
            'hidden_dims': [layer.out_features for layer in self.network 
                          if isinstance(layer, nn.Linear)][:-1],
            'dropout_rate': self.network[3].p  # Assuming dropout is at index 3
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseModel':
        """Create model instance from configuration"""
        return cls(**config)
    
    def save(self, path: str) -> None:
        """Save model weights and configuration"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.get_config()
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model from saved checkpoint"""
        checkpoint = torch.load(path)
        model = cls.from_config(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions for input data"""
        self.eval()
        with torch.no_grad():
            return self.forward(x) 