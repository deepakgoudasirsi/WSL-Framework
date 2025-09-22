import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import torchvision

class BootstrappingLoss(nn.Module):
    """Bootstrapping loss for handling noisy labels"""
    def __init__(self, beta: float = 0.95):
        super().__init__()
        self.beta = beta
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(outputs, dim=1)
        pred_probs, pred_labels = torch.max(probs, dim=1)
        
        # Combine ground truth and predictions
        targets = targets * self.beta + pred_labels * (1 - self.beta)
        targets = targets.long()
        
        # Calculate loss
        loss = self.ce(outputs, targets)
        return loss.mean()

class GCE(nn.Module):
    """Generalized Cross Entropy loss"""
    def __init__(self, q: float = 0.7):
        super().__init__()
        self.q = q
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(outputs, dim=1)
        target_probs = probs[torch.arange(probs.size(0)), targets]
        
        # Add small epsilon to prevent division by zero and log(0)
        eps = 1e-8
        target_probs = torch.clamp(target_probs, min=eps, max=1.0-eps)
        
        # Handle the case where q is very close to 0
        if abs(self.q) < 1e-6:
            # Use standard cross entropy as fallback
            return F.cross_entropy(outputs, targets)
        
        # Use the correct GCE formula
        if self.q == 1.0:
            # When q=1, GCE becomes standard cross entropy
            return F.cross_entropy(outputs, targets)
        else:
            # GCE formula: (1 - p^q) / q
            loss = (1.0 - torch.pow(target_probs, self.q)) / self.q
            return loss.mean()

class SCE(nn.Module):
    """Symmetric Cross Entropy loss"""
    def __init__(self, alpha: float = 0.1, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE loss
        ce_loss = self.ce(outputs, targets)
        
        # RCE loss with proper implementation
        probs = F.softmax(outputs, dim=1)
        eps = 1e-8
        probs = torch.clamp(probs, min=eps, max=1.0-eps)
        
        # Create one-hot encoded targets
        num_classes = outputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        # RCE: -sum(p * log(q)) where p is true distribution, q is predicted
        rce_loss = -torch.sum(targets_one_hot * torch.log(probs), dim=1).mean()
        
        # Check for NaN or Inf values
        if torch.isnan(rce_loss) or torch.isinf(rce_loss):
            return ce_loss
        
        return self.alpha * ce_loss + self.beta * rce_loss

class ForwardCorrection(nn.Module):
    """Forward correction for handling noisy labels"""
    def __init__(self, num_classes: int, transition_matrix: torch.Tensor):
        super().__init__()
        self.num_classes = num_classes
        self.transition_matrix = transition_matrix
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply transition matrix to outputs
        corrected_outputs = torch.matmul(outputs, self.transition_matrix)
        return self.ce(corrected_outputs, targets)

class RobustCNN(nn.Module):
    """CNN model with noise-robust training capabilities"""
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,  # Changed from 1 to 3 for CIFAR-10
        loss_type: str = 'gce'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.q = 0.7  # Default q value for GCE
        self.alpha = 0.1  # Default alpha for SCE
        self.beta = 1.0  # Default beta for SCE
        self.current_epoch = 0  # Track current epoch for hybrid training
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.gce_loss = GCE(q=self.q)
        self.sce_loss = SCE(alpha=self.alpha, beta=self.beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Compute noise-robust loss with hybrid training"""
        # Use standard CE loss for first 5 epochs to stabilize training
        if self.current_epoch < 5:
            return self.ce_loss(outputs, targets)
        
        # After 5 epochs, use the specified robust loss
        if self.loss_type == 'gce':
            return self.gce_loss(outputs, targets)
        elif self.loss_type == 'sce':
            return self.sce_loss(outputs, targets)
        elif self.loss_type == 'forward':
            return self._forward_correction(outputs, targets, reduction)
        else:
            # Fallback to standard CE
            return self.ce_loss(outputs, targets)
    
    def set_epoch(self, epoch: int):
        """Set current epoch for hybrid training"""
        self.current_epoch = epoch
    
    def _generalized_cross_entropy(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Generalized Cross Entropy loss"""
        probs = F.softmax(outputs, dim=1)
        probs = probs[torch.arange(len(targets)), targets]
        
        # Add numerical stability
        eps = 1e-8
        probs = torch.clamp(probs, min=eps, max=1.0-eps)
        
        # Handle the case where q is very close to 0
        if abs(self.q) < 1e-6:
            return F.cross_entropy(outputs, targets)
        
        # Use the correct GCE formula
        if self.q == 1.0:
            return F.cross_entropy(outputs, targets)
        else:
            loss = (1.0 - torch.pow(probs, self.q)) / self.q
            return loss.mean() if reduction == 'mean' else loss
    
    def _symmetric_cross_entropy(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Symmetric Cross Entropy loss"""
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        rce_loss = self._reverse_cross_entropy(outputs, targets)
        loss = ce_loss + self.alpha * rce_loss
        return loss.mean() if reduction == 'mean' else loss
    
    def _reverse_cross_entropy(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Reverse Cross Entropy component"""
        probs = F.softmax(outputs, dim=1)
        probs = probs[torch.arange(len(targets)), targets]
        return -torch.log(1 - probs + 1e-8)
    
    def _forward_correction(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Forward Correction loss"""
        # Assuming uniform noise transition matrix
        T = torch.ones(self.num_classes, self.num_classes) * (self.beta / (self.num_classes - 1))
        T.fill_diagonal_(1 - self.beta)
        T = T.to(outputs.device)
        
        # Forward correction
        probs = F.softmax(outputs, dim=1)
        corrected_probs = torch.matmul(probs, T)
        loss = -torch.log(corrected_probs[torch.arange(len(targets)), targets])
        return loss.mean() if reduction == 'mean' else loss

class RobustResNet(nn.Module):
    """Noise-robust ResNet model"""
    def __init__(
        self,
        num_classes: int = 10,
        model_type: str = 'resnet18',
        loss_type: str = 'gce',
        q: float = 0.7,
        alpha: float = 0.1,
        beta: float = 1.0,
        transition_matrix: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        # Use custom ResNet implementation to avoid TorchVision multiprocessing issues
        if model_type == 'resnet18':
            self.model = custom_resnet18(num_classes)
        elif model_type == 'resnet50':
            self.model = custom_resnet50(num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Loss function parameters
        self.loss_type = loss_type
        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.current_epoch = 0  # Track current epoch for hybrid training
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.gce_loss = GCE(q=q)
        self.sce_loss = SCE(alpha=alpha, beta=beta)
        
        if loss_type == 'forward':
            if transition_matrix is None:
                raise ValueError("Transition matrix required for forward correction")
            self.forward_loss = ForwardCorrection(num_classes, transition_matrix)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Compute noise-robust loss with hybrid training"""
        # Use standard CE loss for first 5 epochs to stabilize training
        if self.current_epoch < 5:
            loss = self.ce_loss(outputs, targets)
        else:
            # After 5 epochs, use the specified robust loss
            if self.loss_type == 'gce':
                loss = self.gce_loss(outputs, targets)
            elif self.loss_type == 'sce':
                loss = self.sce_loss(outputs, targets)
            elif self.loss_type == 'forward':
                loss = self.forward_loss(outputs, targets)
            else:
                # Fallback to standard CE
                loss = self.ce_loss(outputs, targets)
        
        pred = outputs.argmax(dim=1)
        acc = (pred == targets).float().mean().item()
        return loss, acc
    
    def set_epoch(self, epoch: int):
        """Set current epoch for hybrid training"""
        self.current_epoch = epoch

class BasicBlock(nn.Module):
    """Basic ResNet block"""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CustomResNet(nn.Module):
    """Custom ResNet implementation without TorchVision dependencies"""
    def __init__(self, block, num_blocks, num_classes=10):
        super(CustomResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def custom_resnet18(num_classes=10):
    """Custom ResNet-18 without TorchVision"""
    return CustomResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def custom_resnet50(num_classes=10):
    """Custom ResNet-50 without TorchVision"""
    return CustomResNet(BasicBlock, [3, 4, 6, 3], num_classes) 