#!/usr/bin/env python3
"""
Unified WSL Experiment
Combines multiple WSL strategies in a single experiment
"""

# Import warning suppression utility first
from src.utils.warning_suppression import apply_warning_suppression
apply_warning_suppression()

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from src.models.semi_supervised import SemiSupervisedModel
# Import local modules
from src.models.baseline import SimpleCNN, ResNet, MLP
from src.data.preprocessing import get_cifar10_dataloaders, get_mnist_dataloaders
from src.data.clothing1m import get_clothing1m_dataloaders
from src.models.noise_robust import RobustCNN, RobustResNet
from src.training.train import Trainer
from src.utils.visualization import plot_training_curves

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_semi_supervised_dataset(
    dataset: Dataset,
    labeled_ratio: float = 0.1,
    num_classes: int = 10
) -> Tuple[Dataset, Dataset]:
    """Create labeled and unlabeled datasets for semi-supervised learning"""
    num_samples = len(dataset)
    num_labeled = int(num_samples * labeled_ratio)
    num_unlabeled = num_samples - num_labeled
    
    # Split dataset into labeled and unlabeled
    labeled_dataset, unlabeled_dataset = random_split(
        dataset, [num_labeled, num_unlabeled]
    )
    
    return labeled_dataset, unlabeled_dataset

def get_cifar10_dataloaders(
    batch_size: int,
    labeled_ratio: float = 0.1,
    num_workers: int = 0  # Set to 0 to avoid subprocess warnings
) -> Dict[str, DataLoader]:
    """Get CIFAR-10 dataloaders for semi-supervised learning"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('data', train=False, transform=transform_test)
    
    labeled_dataset, unlabeled_dataset = create_semi_supervised_dataset(
        train_dataset, labeled_ratio
    )
    
    labeled_loader = DataLoader(
        labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return {
        'labeled': labeled_loader,
        'unlabeled': unlabeled_loader,
        'test': test_loader
    }

def get_mnist_dataloaders(
    batch_size: int,
    labeled_ratio: float = 0.1,
    num_workers: int = 0  # Set to 0 to avoid subprocess warnings
) -> Dict[str, DataLoader]:
    """Get MNIST dataloaders for semi-supervised learning"""
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST('data', train=False, transform=transform_test)
    
    labeled_dataset, unlabeled_dataset = create_semi_supervised_dataset(
        train_dataset, labeled_ratio
    )
    
    labeled_loader = DataLoader(
        labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return {
        'labeled': labeled_loader,
        'unlabeled': unlabeled_loader,
        'test': test_loader
    }

class UnifiedWSLExperiment:
    """Unified WSL experiment runner that combines multiple strategies"""
    
    def __init__(
        self,
        dataset: str,
        model_type: str,
        strategies: List[str],
        labeled_ratio: float = 0.1,
        batch_size: int = 128,
        learning_rate: float = 0.0005,  # Reduced learning rate for stability
        weight_decay: float = 1e-4,
        epochs: int = 100,
        noise_rate: float = 0.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.dataset = dataset
        self.model_type = model_type
        self.strategies = strategies
        self.labeled_ratio = labeled_ratio
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.noise_rate = noise_rate
        self.device = device
        
        # Create experiment directory
        self.experiment_dir = self._create_experiment_dir()
        
        # Initialize model and dataloaders
        self.dataloaders = self._get_dataloaders()
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize metrics
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'combined_score': []
        }
        
        self.best_val_acc = 0
        self.best_epoch = 0
    
    def _create_experiment_dir(self) -> Path:
        """Create directory for experiment results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        strategies_str = '_'.join(self.strategies)
        experiment_name = f"{self.dataset}_{self.model_type}_{strategies_str}_{timestamp}"
        experiment_dir = Path('experiments/unified_wsl') / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir
    
    def _get_dataloaders(self) -> Dict[str, DataLoader]:
        """Get dataloaders for the specified dataset"""
        if self.dataset == 'cifar10':
            return get_cifar10_dataloaders(
                batch_size=self.batch_size,
                labeled_ratio=self.labeled_ratio
            )
        elif self.dataset == 'mnist':
            return get_mnist_dataloaders(
                batch_size=self.batch_size,
                labeled_ratio=self.labeled_ratio
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
    
    def _create_model(self) -> nn.Module:
        """Create the model with proper initialization"""
        if self.model_type == 'mlp':
            if self.dataset == 'mnist':
                model = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(784, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 10)
                )
            else:  # CIFAR-10
                model = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(3072, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 10)
                )
        elif self.model_type == 'cnn':
            model = nn.Sequential(
                nn.Conv2d(3 if self.dataset == 'cifar10' else 1, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 10)
            )
        elif self.model_type == 'resnet':
            # Create a proper ResNet implementation
            model = self._create_resnet()
        elif self.model_type == 'simple_cnn':
            model = SimpleCNN(num_classes=10)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Initialize weights with better numerical stability
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # Reduced gain for stability
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        model.apply(init_weights)
        return model
    
    def _create_resnet(self) -> nn.Module:
        """Create a ResNet model suitable for CIFAR-10/MNIST"""
        class BasicBlock(nn.Module):
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
        
        class ResNet(nn.Module):
            def __init__(self, block, num_blocks, num_classes=10, input_channels=3):
                super(ResNet, self).__init__()
                self.in_planes = 64
                
                self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
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
        
        # Create ResNet-18
        input_channels = 3 if self.dataset == 'cifar10' else 1
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, input_channels=input_channels)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with better numerical stability"""
        # Use different learning rates for different model types
        if self.model_type == 'resnet':
            # Higher learning rate for ResNet
            lr = self.learning_rate * 10  # 0.001 for ResNet
        else:
            lr = self.learning_rate
        
        # Use a more conservative learning rate for stability
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.weight_decay,
            eps=1e-8,  # Higher epsilon for numerical stability
            betas=(0.9, 0.999)
        )
        return optimizer
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        # Use different schedulers for different model types
        if self.model_type == 'resnet':
            # Step scheduler for ResNet
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            # ReduceLROnPlateau for other models
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True,
                min_lr=1e-7
            )
        return scheduler
    
    def _save_config(self):
        """Save experiment configuration"""
        config = {
            'dataset': self.dataset,
            'model_type': self.model_type,
            'strategies': self.strategies,
            'labeled_ratio': self.labeled_ratio,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'noise_rate': self.noise_rate,
            'device': self.device
        }
        
        with open(self.experiment_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
    
    def _save_results(self):
        """Save experiment results"""
        results = {
            'metrics': self.metrics,
            'best_epoch': self.best_epoch,
            'best_val_acc': self.best_val_acc
        }
        
        with open(self.experiment_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=4)
    
    def _plot_training_curves(self):
        """Plot and save training curves"""
        plot_training_curves(
            self.metrics,
            save_path=self.experiment_dir / 'training_curves.png'
        )
    
    def _augment_data(self, data):
        """Apply data augmentation"""
        if self.dataset == 'cifar10':
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                data = torch.flip(data, [3])
            # Random crop
            if torch.rand(1) > 0.5:
                data = F.pad(data, (2, 2, 2, 2), mode='reflect')
                data = F.interpolate(data, size=(32, 32), mode='bilinear', align_corners=False)
        elif self.dataset == 'mnist':
            # Simple augmentations for MNIST
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                data = torch.flip(data, [3])
            # Random vertical flip
            if torch.rand(1) > 0.5:
                data = torch.flip(data, [2])
            # Add small random noise
            if torch.rand(1) > 0.5:
                noise = torch.randn_like(data) * 0.1
                data = data + noise
                data = torch.clamp(data, 0, 1)
        
        return data
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch using combined strategies"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        combined_score = 0
        num_batches = 0
        
        labeled_loader = self.dataloaders['labeled']
        unlabeled_loader = self.dataloaders['unlabeled']
        
        # Create loss function once
        criterion = nn.CrossEntropyLoss()
        
        # Train on labeled data
        for batch_idx, (data, target) in enumerate(labeled_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected at batch {batch_idx}")
                continue
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            num_batches += 1
        
        # Apply unified WSL strategies on unlabeled data
        strategy_scores = []
        
        # Apply all strategies together in a unified manner
        if self.strategies:
            unified_score = self._apply_unified_strategies(unlabeled_loader)
            strategy_scores.append(unified_score)
        
        # Calculate combined score
        if strategy_scores:
            combined_score = np.mean(strategy_scores)
        
        # Handle division by zero
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = 100. * correct / max(total, 1)
        
        return {
            'loss': avg_loss,
            'acc': accuracy,
            'combined_score': combined_score
        }
    
    def _apply_unified_strategies(self, unlabeled_loader) -> float:
        """Apply all strategies together in a unified manner"""
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(unlabeled_loader):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Initialize combined loss
            combined_loss = 0
            num_strategies = 0
            
            # Apply consistency regularization
            if 'consistency' in self.strategies:
                aug1 = self._augment_data(data)
                aug2 = self._augment_data(data)
                output1 = self.model(aug1)
                output2 = self.model(aug2)
                
                # Use KL divergence instead of MSE for better stability
                cons_loss = F.kl_div(
                    F.log_softmax(output1, dim=1),
                    F.softmax(output2, dim=1),
                    reduction='batchmean'
                )
                
                # Check for valid loss
                if not torch.isnan(cons_loss) and not torch.isinf(cons_loss):
                    combined_loss += cons_loss
                    num_strategies += 1
            
            # Apply pseudo-labeling
            if 'pseudo_label' in self.strategies:
                confidence_threshold = 0.95
                output = self.model(data)
                probs = F.softmax(output, dim=1)
                confidence, pseudo_labels = torch.max(probs, dim=1)
                
                # Only use high-confidence predictions
                mask = confidence > confidence_threshold
                if mask.sum() > 0:
                    pseudo_loss = F.cross_entropy(output[mask], pseudo_labels[mask])
                    
                    # Check for valid loss
                    if not torch.isnan(pseudo_loss) and not torch.isinf(pseudo_loss):
                        combined_loss += pseudo_loss
                        num_strategies += 1
            
            # Apply co-training (simplified version)
            if 'co_training' in self.strategies:
                # Use different augmentations for co-training
                aug1 = self._augment_data(data)
                aug2 = self._augment_data(data)
                output1 = self.model(aug1)
                output2 = self.model(aug2)
                
                # Use symmetric KL divergence for better stability
                co_loss1 = F.kl_div(
                    F.log_softmax(output1, dim=1),
                    F.softmax(output2, dim=1),
                    reduction='batchmean'
                )
                co_loss2 = F.kl_div(
                    F.log_softmax(output2, dim=1),
                    F.softmax(output1, dim=1),
                    reduction='batchmean'
                )
                co_loss = (co_loss1 + co_loss2) / 2
                
                # Check for valid loss
                if not torch.isnan(co_loss) and not torch.isinf(co_loss):
                    combined_loss += co_loss
                    num_strategies += 1
            
            # Average the losses only if we have valid strategies
            if num_strategies > 0:
                combined_loss = combined_loss / num_strategies
                
                # Additional safety check
                if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                    print(f"Warning: Invalid loss detected at batch {batch_idx}, skipping...")
                    continue
                
                # Gradient clipping before backward pass
                combined_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += combined_loss.item()
                num_batches += 1
            else:
                print(f"Warning: No valid strategies applied at batch {batch_idx}")
        
        # Return average loss, or 0 if no valid batches
        return total_loss / max(num_batches, 1)
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.dataloaders['test']:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected in validation")
                    continue
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                num_batches += 1
        
        # Handle division by zero
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = 100. * correct / max(total, 1)
        
        return {
            'loss': avg_loss,
            'acc': accuracy
        }
    
    def run(self) -> Dict[str, Any]:
        """Run the experiment"""
        print(f"Starting unified WSL experiment: {self.experiment_dir}")
        print(f"Strategies: {self.strategies}")
        print(f"Dataset: {self.dataset}, Model: {self.model_type}")
        print(f"Labeled ratio: {self.labeled_ratio}, Epochs: {self.epochs}")
        print(f"Device: {self.device}")
        print("-" * 60)
        
        self._save_config()
        
        best_val_acc = 0
        best_epoch = 0
        
        try:
            for epoch in range(self.epochs):
                # Train
                train_metrics = self.train_epoch()
                
                # Validate
                val_metrics = self.validate_epoch()
                
                # Update learning rate
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['acc'])
                else:
                    self.scheduler.step()
                
                # Store metrics
                self.metrics['train_loss'].append(train_metrics['loss'])
                self.metrics['train_acc'].append(train_metrics['acc'])
                self.metrics['val_loss'].append(val_metrics['loss'])
                self.metrics['val_acc'].append(val_metrics['acc'])
                self.metrics['combined_score'].append(train_metrics.get('combined_score', 0))
                
                # Save best model
                if val_metrics['acc'] > best_val_acc:
                    best_val_acc = val_metrics['acc']
                    best_epoch = epoch
                    torch.save(
                        self.model.state_dict(),
                        self.experiment_dir / 'best_model.pth'
                    )
                
                # Print progress with additional metrics
                print(f"Epoch {epoch+1}/{self.epochs}")
                print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.2f}%")
                print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.2f}%")
                if 'combined_score' in train_metrics:
                    print(f"Combined Score: {train_metrics['combined_score']:.3f}")
                print(f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch+1})")
                print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                print("-" * 50)
                
                # Save intermediate results every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self._save_results()
                    print(f"Intermediate results saved at epoch {epoch+1}")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving current results...")
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
            print("Saving current results...")
        
        # Save final results and plots
        self._save_results()
        self._plot_training_curves()
        
        print(f"\nExperiment completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")
        
        return {
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'final_metrics': self.metrics
        }

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Unified WSL Experiments')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                       help='Dataset to use (default: mnist)')
    parser.add_argument('--model_type', type=str, default='mlp', 
                       choices=['mlp', 'simple_cnn', 'resnet'],
                       help='Model type to use (default: mlp)')
    parser.add_argument('--strategies', nargs='+', default=['pseudo_label'],
                       choices=['pseudo_label', 'consistency', 'co_training'],
                       help='WSL strategies to combine (default: pseudo_label)')
    parser.add_argument('--labeled_ratio', type=float, default=0.1,
                       help='Ratio of labeled data (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--noise_rate', type=float, default=0.0,
                       help='Noise rate for robust training (default: 0.0)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Run experiment with macOS warning suppression
    experiment = UnifiedWSLExperiment(
        dataset=args.dataset,
        model_type=args.model_type,
        strategies=args.strategies,
        labeled_ratio=args.labeled_ratio,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        noise_rate=args.noise_rate,
        device=device
    )
    
    results = experiment.run()
    
    print(f"\nExperiment completed!")
    print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Strategies used: {args.strategies}")

if __name__ == '__main__':
    main() 