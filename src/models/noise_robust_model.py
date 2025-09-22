import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from src.utils.data import mixup_data

from .baseline import SimpleCNN, ResNet
from .losses import NoiseRobustLoss, CoTeachingLoss

class NoiseRobustModel(nn.Module):
    """Model with noise-robust training strategies."""
    
    def __init__(
        self,
        model_type: str = 'simple_cnn',
        num_classes: int = 10,
        loss_type: str = 'cross_entropy',
        beta: float = 0.95,
        use_co_teaching: bool = False,
        use_mixup: bool = False,
        use_label_smoothing: bool = False,
        label_smoothing: float = 0.1,
        forget_rate: float = 0.2
    ):
        super().__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.beta = beta
        self.use_co_teaching = use_co_teaching
        self.use_mixup = use_mixup
        self.use_label_smoothing = use_label_smoothing
        self.label_smoothing = label_smoothing
        self.forget_rate = forget_rate
        
        # Create main model
        if model_type == 'simple_cnn':
            self.model1 = SimpleCNN(num_classes=num_classes)
        elif model_type == 'resnet':
            self.model1 = ResNet(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create second model for co-teaching
        if use_co_teaching:
            if model_type == 'simple_cnn':
                self.model2 = SimpleCNN(num_classes=num_classes)
            elif model_type == 'resnet':
                self.model2 = ResNet(num_classes=num_classes)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model1(x)
    
    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        epoch: Optional[int] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Compute loss with selected strategies."""
        if self.use_mixup:
            # Apply mixup
            mixed_x, y_a, y_b, lam = mixup_data(outputs, targets)
            loss = lam * F.cross_entropy(mixed_x, y_a) + (1 - lam) * F.cross_entropy(mixed_x, y_b)
        elif self.use_label_smoothing:
            # Apply label smoothing
            loss = self._label_smoothing_loss(outputs, targets)
        elif self.loss_type == 'bootstrap':
            # Bootstrap loss
            loss = self._bootstrap_loss(outputs, targets)
        else:
            # Standard cross entropy
            loss = F.cross_entropy(outputs, targets)
        
        return loss, {}
    
    def _bootstrap_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute bootstrap loss."""
        probs = F.softmax(outputs, dim=1)
        with torch.no_grad():
            bootstrap_targets = self.beta * F.one_hot(targets, self.num_classes) + \
                              (1 - self.beta) * probs
        return -torch.sum(bootstrap_targets * F.log_softmax(outputs, dim=1), dim=1).mean()
    
    def co_teaching_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        epoch: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one step of co-teaching."""
        # Forward pass through both models
        outputs1 = self.model1(inputs)
        outputs2 = self.model2(inputs)
        
        # Compute losses
        loss1 = F.cross_entropy(outputs1, targets, reduction='none')
        loss2 = F.cross_entropy(outputs2, targets, reduction='none')
        
        # Select small-loss samples
        _, idx1 = torch.sort(loss1)
        _, idx2 = torch.sort(loss2)
        
        # Calculate number of samples to keep
        num_keep = int(len(targets) * (1 - min(epoch / 10, self.forget_rate)))
        
        # Keep only small-loss samples
        idx1 = idx1[:num_keep]
        idx2 = idx2[:num_keep]
        
        # Compute final losses
        loss1 = loss1[idx2].mean()
        loss2 = loss2[idx1].mean()
        
        return loss1, loss2
    
    def _label_smoothing_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute label smoothing loss."""
        log_probs = F.log_softmax(outputs, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(outputs)
            true_dist.fill_(self.label_smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get model state dict"""
        if self.use_co_teaching:
            return {
                'model1': self.model1.state_dict(),
                'model2': self.model2.state_dict()
            }
        return {'model1': self.model1.state_dict()}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load model state dict"""
        if self.use_co_teaching:
            self.model1.load_state_dict(state_dict['model1'])
            self.model2.load_state_dict(state_dict['model2'])
        else:
            self.model1.load_state_dict(state_dict['model1']) 