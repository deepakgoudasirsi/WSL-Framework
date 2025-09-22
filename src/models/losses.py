import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class BootstrappingLoss(nn.Module):
    """Bootstrapping loss for handling noisy labels"""
    def __init__(self, beta: float = 0.95):
        super().__init__()
        self.beta = beta
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Model predictions (N x C)
            target: Target labels (N)
        Returns:
            Loss value
        """
        # Get soft predictions
        pred_soft = F.softmax(pred, dim=1)
        
        # Create one-hot target
        target_one_hot = F.one_hot(target, pred.size(1)).float()
        
        # Bootstrap target
        bootstrap_target = self.beta * target_one_hot + (1 - self.beta) * pred_soft
        
        # Compute cross entropy
        loss = -torch.sum(bootstrap_target * F.log_softmax(pred, dim=1), dim=1).mean()
        return loss

class ForwardCorrectionLoss(nn.Module):
    """Forward correction loss for handling noisy labels"""
    def __init__(self, transition_matrix: torch.Tensor):
        super().__init__()
        self.transition_matrix = transition_matrix
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Model predictions (N x C)
            target: Target labels (N)
        Returns:
            Loss value
        """
        # Get soft predictions
        pred_soft = F.softmax(pred, dim=1)
        
        # Create one-hot target
        target_one_hot = F.one_hot(target, pred.size(1)).float()
        
        # Apply transition matrix
        corrected_target = torch.matmul(target_one_hot, self.transition_matrix)
        
        # Compute cross entropy
        loss = -torch.sum(corrected_target * F.log_softmax(pred, dim=1), dim=1).mean()
        return loss

class CoTeachingLoss(nn.Module):
    """Co-teaching loss for handling noisy labels"""
    def __init__(self, forget_rate: float = 0.2):
        super().__init__()
        self.forget_rate = forget_rate
        
    def forward(
        self,
        pred1: torch.Tensor,
        pred2: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pred1: Predictions from first model (N x C)
            pred2: Predictions from second model (N x C)
            target: Target labels (N)
        Returns:
            Tuple of (loss1, loss2)
        """
        # Get predictions
        pred1_soft = F.softmax(pred1, dim=1)
        pred2_soft = F.softmax(pred2, dim=1)
        
        # Create one-hot target
        target_one_hot = F.one_hot(target, pred1.size(1)).float()
        
        # Compute losses
        loss1 = -torch.sum(target_one_hot * F.log_softmax(pred1, dim=1), dim=1)
        loss2 = -torch.sum(target_one_hot * F.log_softmax(pred2, dim=1), dim=1)
        
        # Select samples with small loss
        _, idx1 = torch.sort(loss1)
        _, idx2 = torch.sort(loss2)
        
        num_forget = int(self.forget_rate * len(loss1))
        idx1 = idx1[num_forget:]
        idx2 = idx2[num_forget:]
        
        # Compute final losses
        loss1 = loss1[idx2].mean()
        loss2 = loss2[idx1].mean()
        
        return loss1, loss2

class DynamicBootstrappingLoss(nn.Module):
    """Dynamic bootstrapping loss that adapts to noise level"""
    def __init__(self, num_classes: int, momentum: float = 0.9):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.register_buffer('class_weights', torch.ones(num_classes))
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Model predictions (N x C)
            target: Target labels (N)
        Returns:
            Loss value
        """
        # Get soft predictions
        pred_soft = F.softmax(pred, dim=1)
        
        # Create one-hot target
        target_one_hot = F.one_hot(target, self.num_classes).float()
        
        # Update class weights
        with torch.no_grad():
            class_counts = target_one_hot.sum(dim=0)
            class_weights = 1.0 / (class_counts + 1e-6)
            class_weights = class_weights / class_weights.sum()
            self.class_weights = self.momentum * self.class_weights + (1 - self.momentum) * class_weights
        
        # Compute weighted loss
        loss = -torch.sum(
            target_one_hot * F.log_softmax(pred, dim=1) * self.class_weights,
            dim=1
        ).mean()
        
        return loss

class NoiseRobustLoss(nn.Module):
    """Combined noise-robust loss function"""
    def __init__(
        self,
        loss_type: str = 'bootstrap',
        beta: float = 0.95,
        transition_matrix: Optional[torch.Tensor] = None,
        num_classes: int = 10
    ):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'bootstrap':
            self.loss_fn = BootstrappingLoss(beta=beta)
        elif loss_type == 'forward':
            if transition_matrix is None:
                raise ValueError("Transition matrix required for forward correction")
            self.loss_fn = ForwardCorrectionLoss(transition_matrix)
        elif loss_type == 'dynamic':
            self.loss_fn = DynamicBootstrappingLoss(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target) 