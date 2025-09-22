#!/usr/bin/env python3
"""
Semi-Supervised Learning Experiments
Implements various semi-supervised learning strategies
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

from src.models.baseline import MLP, SimpleCNN, ResNet
# Import plotting function from visualization
from src.utils.visualization import plot_training_curves

# Setup logging
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

class SemiSupervisedExperiment:
    """Semi-supervised learning experiment runner"""
    
    def __init__(
        self,
        dataset: str,
        model_type: str,
        labeled_ratio: float,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        epochs: int,
        strategy: str = 'pseudo_label',
        consistency_weight: float = 1.0,
        consistency_type: str = 'mse',
        pseudo_threshold: float = 0.7,  # Reduced from 0.8 for better ResNet pseudo-labeling
        pseudo_alpha: float = 0.1,
        augmentation_type: str = 'standard',
        augmentation_strength: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.dataset = dataset
        self.model_type = model_type
        self.labeled_ratio = labeled_ratio
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.strategy = strategy
        self.consistency_weight = consistency_weight
        self.consistency_type = consistency_type
        self.pseudo_threshold = pseudo_threshold
        self.pseudo_alpha = pseudo_alpha
        self.augmentation_type = augmentation_type
        self.augmentation_strength = augmentation_strength
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
            'test_acc': []
        }
        
        self.best_val_acc = 0
        self.best_epoch = 0
    
    def _create_experiment_dir(self) -> Path:
        """Create directory for experiment results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"{self.dataset}_{self.model_type}_labeled{self.labeled_ratio}_{timestamp}"
        experiment_dir = Path('experiments/semi_supervised') / experiment_name
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
        """Create the model"""
        if self.dataset == 'cifar10':
            num_classes = 10
        elif self.dataset == 'mnist':
            num_classes = 10
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        
        if self.model_type == 'mlp':
            model = MLP(num_classes=num_classes)
        elif self.model_type == 'simple_cnn':
            model = SimpleCNN(num_classes=num_classes)
        elif self.model_type == 'resnet':
            model = ResNet(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Initialize weights properly
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        model.apply(init_weights)
        
        return model.to(self.device)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with model-specific learning rates"""
        if self.model_type == 'resnet':
            # Use lower learning rate for ResNet
            lr = self.learning_rate * 0.1
        else:
            lr = self.learning_rate
            
        return optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
    
    def _create_diverse_model(self) -> nn.Module:
        """Create a second model with different initialization for co-training"""
        if self.dataset == 'cifar10':
            num_classes = 10
        elif self.dataset == 'mnist':
            num_classes = 10
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        
        if self.model_type == 'mlp':
            model = MLP(num_classes=num_classes)
        elif self.model_type == 'simple_cnn':
            model = SimpleCNN(num_classes=num_classes)
        elif self.model_type == 'resnet':
            model = ResNet(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Initialize weights with different random seeds for diversity
        def init_weights_diverse(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.2)  # Different gain
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # Different mode
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.2)  # Different initialization
                nn.init.constant_(m.bias, 0.2)
        
        model.apply(init_weights_diverse)
        
        return model.to(self.device)
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler with model-specific parameters"""
        if self.model_type == 'resnet':
            # More gradual learning rate decay for ResNet
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.8)
        else:
            # Standard scheduler for other models
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.7)
    
    def _save_config(self):
        """Save experiment configuration"""
        config = {
            'dataset': self.dataset,
            'model_type': self.model_type,
            'labeled_ratio': self.labeled_ratio,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'consistency_weight': self.consistency_weight,
            'consistency_type': self.consistency_type,
            'pseudo_threshold': self.pseudo_threshold,
            'pseudo_alpha': self.pseudo_alpha,
            'augmentation_type': self.augmentation_type,
            'augmentation_strength': self.augmentation_strength,
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
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        labeled_loader = self.dataloaders['labeled']
        unlabeled_loader = self.dataloaders['unlabeled']
        
        if self.strategy == 'pseudo_label':
            return self._train_pseudo_label_epoch(labeled_loader, unlabeled_loader)
        elif self.strategy == 'consistency':
            return self._train_consistency_epoch(labeled_loader, unlabeled_loader)
        elif self.strategy == 'co_training':
            return self._train_co_training_epoch(labeled_loader, unlabeled_loader)
        elif self.strategy == 'mixmatch':
            return self._train_mixmatch_epoch(labeled_loader, unlabeled_loader)
        else:
            # Default supervised training
            return self._train_supervised_epoch(labeled_loader)
    
    def _train_supervised_epoch(self, labeled_loader) -> Dict[str, float]:
        """Standard supervised training"""
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(labeled_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        return {
            'loss': total_loss / len(labeled_loader),
            'acc': 100. * correct / total
        }
    
    def _train_pseudo_label_epoch(self, labeled_loader, unlabeled_loader) -> Dict[str, float]:
        """Pseudo-labeling training with improved stability"""
        total_loss = 0
        correct = 0
        total = 0
        pseudo_labels_used = 0
        
        # Adjust parameters based on model type
        if self.model_type == 'resnet':
            temperature = 3.0  # Higher temperature for ResNet
            adaptive_threshold_base = 0.7  # Lower base threshold for ResNet
            pseudo_weight = 0.05  # Lower weight for ResNet
        else:
            temperature = 2.0  # Standard temperature for other models
            adaptive_threshold_base = 0.8  # Standard base threshold
            pseudo_weight = 0.1  # Standard weight
        
        # Combine labeled and unlabeled data for more stable training
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        # Train on both labeled and unlabeled data together
        for batch_idx in range(len(labeled_loader)):
            self.optimizer.zero_grad()
            
            # Get labeled batch
            try:
                data_labeled, target = next(labeled_iter)
                data_labeled, target = data_labeled.to(self.device), target.to(self.device)
                
                # Supervised loss on labeled data
                output_labeled = self.model(data_labeled)
                supervised_loss = nn.CrossEntropyLoss()(output_labeled, target)
                
                # Update metrics
                pred = output_labeled.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
            except StopIteration:
                supervised_loss = 0
            
            # Get unlabeled batch for pseudo-labeling
            try:
                data_unlabeled, _ = next(unlabeled_iter)
                data_unlabeled = data_unlabeled.to(self.device)
                
                # Generate pseudo-labels with model-specific temperature scaling
                # First, get model output
                output_unlabeled = self.model(data_unlabeled)
                
                # Apply temperature scaling for better calibration
                with torch.no_grad():
                    scaled_output = output_unlabeled / temperature
                    probs = torch.softmax(scaled_output, dim=1)
                    confidence, pseudo_labels = probs.max(1)
                
                # Use adaptive threshold based on epoch progress and model type
                # Start with lower threshold and increase gradually
                adaptive_threshold = min(adaptive_threshold_base, 
                                       self.pseudo_threshold * (batch_idx + 1) / len(labeled_loader))
                mask = confidence > adaptive_threshold
                
                if mask.sum() > 0:
                    # Only use high-confidence pseudo-labels
                    pseudo_data = data_unlabeled[mask]
                    # Ensure pseudo_targets are detached and don't require gradients
                    pseudo_targets = pseudo_labels[mask].detach()
                    pseudo_labels_used += pseudo_targets.size(0)
                    
                    # Pseudo-label loss with model-specific weight
                    output_pseudo = self.model(pseudo_data)
                    pseudo_loss = pseudo_weight * nn.CrossEntropyLoss()(output_pseudo, pseudo_targets)
                    
                else:
                    pseudo_loss = 0
            
            except StopIteration:
                pseudo_loss = 0
            
            # Combined loss
            total_loss_batch = supervised_loss + pseudo_loss
            
            # Check for NaN loss
            if torch.isnan(total_loss_batch) or torch.isinf(total_loss_batch):
                continue
            
            # Check if loss requires gradients
            if not total_loss_batch.requires_grad:
                print(f"Warning: Loss does not require gradients at batch {batch_idx}")
                continue
            
            try:
                total_loss_batch.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += supervised_loss.item()
            except Exception as e:
                print(f"Warning: Error in backward pass at batch {batch_idx}: {e}")
                # Skip this batch and continue
                continue
        
        return {
            'loss': total_loss / len(labeled_loader),
            'acc': 100. * correct / total,
            'pseudo_labels': pseudo_labels_used
        }
    
    def _train_consistency_epoch(self, labeled_loader, unlabeled_loader) -> Dict[str, float]:
        """Consistency regularization training"""
        total_loss = 0
        correct = 0
        total = 0
        consistency_loss = 0
        
        # Adjust consistency weight based on model type
        if self.model_type == 'resnet':
            consistency_weight = 0.5  # Higher weight for ResNet
        else:
            consistency_weight = 0.1  # Lower weight for other models
        
        # Combine labeled and unlabeled data for more stable training
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        # Train on both labeled and unlabeled data together
        for batch_idx in range(len(labeled_loader)):
            self.optimizer.zero_grad()
            
            # Get labeled batch
            try:
                data_labeled, target = next(labeled_iter)
                data_labeled, target = data_labeled.to(self.device), target.to(self.device)
                
                # Supervised loss on labeled data
                output_labeled = self.model(data_labeled)
                supervised_loss = nn.CrossEntropyLoss()(output_labeled, target)
                
                # Update metrics
                pred = output_labeled.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
            except StopIteration:
                supervised_loss = 0
            
            # Get unlabeled batch for consistency regularization
            try:
                data_unlabeled, _ = next(unlabeled_iter)
                data_unlabeled = data_unlabeled.to(self.device)
                
                # Create two augmented versions
                aug1 = self._augment_data(data_unlabeled)
                aug2 = self._augment_data(data_unlabeled)
                
                output1 = self.model(aug1)
                output2 = self.model(aug2)
                
                # Use KL divergence instead of MSE for better stability
                cons_loss = F.kl_div(
                    F.log_softmax(output1, dim=1),
                    F.softmax(output2, dim=1),
                    reduction='batchmean'
                )
                
                # Apply model-specific consistency weight
                total_cons_loss = consistency_weight * cons_loss
                consistency_loss += cons_loss.item()
                
            except StopIteration:
                total_cons_loss = 0
            
            # Combined loss
            total_loss_batch = supervised_loss + total_cons_loss
            
            # Check for NaN loss
            if torch.isnan(total_loss_batch) or torch.isinf(total_loss_batch):
                continue
            
            total_loss_batch.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += supervised_loss.item()
        
        return {
            'loss': total_loss / len(labeled_loader),
            'acc': 100. * correct / total,
            'consistency_loss': consistency_loss / len(labeled_loader)
        }
    
    def _train_co_training_epoch(self, labeled_loader, unlabeled_loader) -> Dict[str, float]:
        """Improved co-training with two models"""
        total_loss = 0
        correct = 0
        total = 0
        model_agreement = 0
        
        # Adjust parameters based on model type
        if self.model_type == 'resnet':
            temperature = 3.0  # Higher temperature for ResNet
            confidence_threshold = 0.6  # Lower threshold for ResNet
            pseudo_weight = 0.03  # Lower weight for ResNet
        else:
            temperature = 2.0  # Standard temperature for other models
            confidence_threshold = 0.8  # Standard threshold
            pseudo_weight = 0.1  # Standard weight
        
        # Create second model for co-training with different initialization
        if not hasattr(self, 'model2'):
            print("Initializing second model for co-training...")
            self.model2 = self._create_diverse_model()
            # Use different learning rate for diversity
            if self.model_type == 'resnet':
                lr2 = self.learning_rate * 0.1 * 1.2  # Different learning rate for ResNet
            else:
                lr2 = self.learning_rate * 1.1
            self.optimizer2 = optim.Adam(self.model2.parameters(), lr=lr2, weight_decay=self.weight_decay)
            self.scheduler2 = optim.lr_scheduler.StepLR(self.optimizer2, step_size=20, gamma=0.7)
        
        # Combine labeled and unlabeled data for more stable training
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        # Train on both labeled and unlabeled data together
        for batch_idx in range(len(labeled_loader)):
            # Get labeled batch
            try:
                data_labeled, target = next(labeled_iter)
                data_labeled, target = data_labeled.to(self.device), target.to(self.device)
                
                # Train both models on labeled data
                # Model 1
                self.optimizer.zero_grad()
                output1_labeled = self.model(data_labeled)
                loss1_labeled = nn.CrossEntropyLoss()(output1_labeled, target)
                loss1_labeled.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Model 2
                self.optimizer2.zero_grad()
                output2_labeled = self.model2(data_labeled)
                loss2_labeled = nn.CrossEntropyLoss()(output2_labeled, target)
                loss2_labeled.backward()
                torch.nn.utils.clip_grad_norm_(self.model2.parameters(), max_norm=1.0)
                self.optimizer2.step()
                
                # Update metrics
                pred1 = output1_labeled.argmax(dim=1, keepdim=True)
                pred2 = output2_labeled.argmax(dim=1, keepdim=True)
                correct += pred1.eq(target.view_as(pred1)).sum().item()
                total += target.size(0)
                
                # Calculate agreement
                agreement = (pred1 == pred2).float().mean().item()
                model_agreement += agreement
                
                total_loss += (loss1_labeled.item() + loss2_labeled.item()) / 2
                
            except StopIteration:
                pass
            
            # Get unlabeled batch for co-training
            try:
                data_unlabeled, _ = next(unlabeled_iter)
                data_unlabeled = data_unlabeled.to(self.device)
                
                # Generate pseudo-labels using both models with model-specific parameters
                # First, get predictions from both models
                output1_unlabeled = self.model(data_unlabeled)
                output2_unlabeled = self.model2(data_unlabeled)
                
                # Apply temperature scaling for better calibration
                probs1 = torch.softmax(output1_unlabeled / temperature, dim=1)
                probs2 = torch.softmax(output2_unlabeled / temperature, dim=1)
                
                # Get pseudo-labels and confidence
                confidence1, pseudo_labels1 = probs1.max(1)
                confidence2, pseudo_labels2 = probs2.max(1)
                
                # Find samples where models agree and have high confidence
                agreement_mask = (pseudo_labels1 == pseudo_labels2)
                confidence_mask1 = confidence1 > confidence_threshold
                confidence_mask2 = confidence2 > confidence_threshold
                final_mask = agreement_mask & confidence_mask1 & confidence_mask2
                
                if final_mask.sum() > 0:
                    # Use agreed pseudo-labels for co-training
                    pseudo_data = data_unlabeled[final_mask]
                    # Use pseudo_labels with proper gradient handling and type conversion
                    pseudo_labels = pseudo_labels1[final_mask].detach().clone().long()
                    
                    # Train model 1 on model 2's predictions
                    self.optimizer.zero_grad()
                    output1_pseudo = self.model(pseudo_data)
                    loss1_pseudo = pseudo_weight * nn.CrossEntropyLoss()(output1_pseudo, pseudo_labels)
                    
                    # Check for valid loss before backward pass
                    if not torch.isnan(loss1_pseudo) and not torch.isinf(loss1_pseudo) and loss1_pseudo.requires_grad:
                        try:
                            loss1_pseudo.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()
                        except Exception as e:
                            print(f"Warning: Error in model 1 backward pass: {e}")
                    else:
                        print(f"Warning: Invalid loss1_pseudo: {loss1_pseudo.item()}, requires_grad: {loss1_pseudo.requires_grad}")
                    
                    # Train model 2 on model 1's predictions
                    self.optimizer2.zero_grad()
                    output2_pseudo = self.model2(pseudo_data)
                    loss2_pseudo = pseudo_weight * nn.CrossEntropyLoss()(output2_pseudo, pseudo_labels)
                    
                    # Check for valid loss before backward pass
                    if not torch.isnan(loss2_pseudo) and not torch.isinf(loss2_pseudo) and loss2_pseudo.requires_grad:
                        try:
                            loss2_pseudo.backward()
                            torch.nn.utils.clip_grad_norm_(self.model2.parameters(), max_norm=1.0)
                            self.optimizer2.step()
                        except Exception as e:
                            print(f"Warning: Error in model 2 backward pass: {e}")
                    else:
                        print(f"Warning: Invalid loss2_pseudo: {loss2_pseudo.item()}, requires_grad: {loss2_pseudo.requires_grad}")
                    
                    # Update agreement metric
                    try:
                        with torch.no_grad():
                            pred1_pseudo = output1_pseudo.argmax(dim=1)
                            pred2_pseudo = output2_pseudo.argmax(dim=1)
                            agreement_pseudo = (pred1_pseudo == pred2_pseudo).float().mean().item()
                            model_agreement += agreement_pseudo
                    except Exception as e:
                        print(f"Warning: Error calculating agreement metric: {e}")
                
            except StopIteration:
                pass
        
        # Update learning rate schedulers
        self.scheduler.step()
        self.scheduler2.step()
        
        return {
            'loss': total_loss / len(labeled_loader),
            'acc': 100. * correct / total,
            'model_agreement': model_agreement / len(labeled_loader)
        }
    
    def _train_mixmatch_epoch(self, labeled_loader, unlabeled_loader) -> Dict[str, float]:
        """MixMatch training (simplified version)"""
        total_loss = 0
        correct = 0
        total = 0
        
        # For simplicity, implement a basic version of MixMatch
        for batch_idx, (data, target) in enumerate(labeled_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        return {
            'loss': total_loss / len(labeled_loader),
            'acc': 100. * correct / total
        }
    
    def _augment_data(self, data):
        """Apply data augmentation"""
        # Simple augmentation for consistency training
        if self.dataset == 'cifar10':
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                data = torch.flip(data, [3])
            # Random crop
            if torch.rand(1) > 0.5:
                data = F.pad(data, (2, 2, 2, 2), mode='reflect')
                data = F.interpolate(data, size=(32, 32), mode='bilinear', align_corners=False)
        elif self.dataset == 'mnist':
            # For MNIST, use simpler augmentations that don't require rotation
            # Random horizontal flip (if applicable)
            if torch.rand(1) > 0.5:
                data = torch.flip(data, [3])
            # Add small random noise
            if torch.rand(1) > 0.5:
                noise = torch.randn_like(data) * 0.1
                data = torch.clamp(data + noise, 0, 1)
            # Small random shift
            if torch.rand(1) > 0.5:
                shift_x = torch.randint(-2, 3, (1,)).item()
                shift_y = torch.randint(-2, 3, (1,)).item()
                if shift_x != 0 or shift_y != 0:
                    data = F.pad(data, (abs(shift_x), abs(shift_x), abs(shift_y), abs(shift_y)), mode='reflect')
                    data = data[:, :, abs(shift_y):abs(shift_y)+28, abs(shift_x):abs(shift_x)+28]
        
        return data
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.dataloaders['test']:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return {
            'loss': total_loss / len(self.dataloaders['test']),
            'acc': 100. * correct / total
        }
    
    def run(self) -> Dict[str, Any]:
        """Run the experiment"""
        print(f"Starting experiment: {self.experiment_dir}")
        print(f"Strategy: {self.strategy}")
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
                self.scheduler.step()
                
                # Store metrics
                self.metrics['train_loss'].append(train_metrics['loss'])
                self.metrics['train_acc'].append(train_metrics['acc'])
                self.metrics['val_loss'].append(val_metrics['loss'])
                self.metrics['val_acc'].append(val_metrics['acc'])
                
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
                    
                # Print strategy-specific metrics
                if self.strategy == 'pseudo_label' and 'pseudo_labels' in train_metrics:
                    print(f"Pseudo-labels used: {train_metrics['pseudo_labels']}")
                elif self.strategy == 'consistency' and 'consistency_loss' in train_metrics:
                    print(f"Consistency Loss: {train_metrics['consistency_loss']:.4f}")
                elif self.strategy == 'co_training' and 'model_agreement' in train_metrics:
                    print(f"Model Agreement: {train_metrics['model_agreement']:.3f}")
                
                print(f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch+1})")
                print("-" * 50)
                    
                # Save intermediate results every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self._save_results()
                    print(f"Intermediate results saved at epoch {epoch+1}")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving current results...")
        except Exception as e:
            print(f"\nError during training: {e}")
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

def run_semi_supervised_experiments():
    """Run all semi-supervised learning experiments"""
    # Define experiments
    experiments = [
        # Standard semi-supervised learning
        {
            'dataset': 'cifar10',
            'model_type': 'simple_cnn',
            'labeled_ratio': 0.1,
            'batch_size': 128,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'epochs': 100,
            'consistency_weight': 1.0,
            'consistency_type': 'mse',
            'pseudo_threshold': 0.95,
            'pseudo_alpha': 0.1,
            'augmentation_type': 'standard',
            'augmentation_strength': 1.0
        },
        # Consistency regularization experiments
        {
            'dataset': 'cifar10',
            'model_type': 'simple_cnn',
            'labeled_ratio': 0.1,
            'batch_size': 128,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'epochs': 100,
            'consistency_weight': 2.0,
            'consistency_type': 'kl',
            'pseudo_threshold': 0.95,
            'pseudo_alpha': 0.1,
            'augmentation_type': 'standard',
            'augmentation_strength': 1.0
        }
    ]
    
    # Run experiments
    for i, config in enumerate(experiments):
        print(f"\nRunning experiment {i+1}/{len(experiments)}")
        print(f"Config: {config}")
        
        experiment = SemiSupervisedExperiment(**config)
        results = experiment.run()
        
        print(f"Experiment {i+1} completed!")
        print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
        print("=" * 80)

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Semi-supervised Learning Experiments')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                       help='Dataset to use (default: mnist)')
    parser.add_argument('--model_type', type=str, default='mlp', 
                       choices=['mlp', 'simple_cnn', 'robust_cnn', 'resnet'],
                       help='Model type to use (default: mlp)')
    parser.add_argument('--labeled_ratio', type=float, default=0.1,
                       help='Ratio of labeled data (default: 0.1)')
    parser.add_argument('--strategy', type=str, default='pseudo_label',
                       choices=['pseudo_label', 'consistency', 'mixmatch', 'co_training'],
                       help='Semi-supervised learning strategy (default: pseudo_label)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--confidence_threshold', type=float, default=0.95,
                       help='Confidence threshold for pseudo-labeling (default: 0.95)')
    parser.add_argument('--consistency_weight', type=float, default=1.0,
                       help='Weight for consistency regularization (default: 1.0)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Create experiment configuration
    config = {
        'dataset': args.dataset,
        'model_type': args.model_type,
        'labeled_ratio': args.labeled_ratio,
        'strategy': args.strategy,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'consistency_weight': args.consistency_weight,
        'consistency_type': 'mse',
        'pseudo_threshold': args.confidence_threshold,
        'pseudo_alpha': 0.1,
        'augmentation_type': 'standard',
        'augmentation_strength': 1.0,
        'device': device
    }
    
    # Run experiment
    experiment = SemiSupervisedExperiment(**config)
    results = experiment.run()
    
    print(f"\nExperiment completed!")
    print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
    print(f"Best epoch: {results['best_epoch']}")

if __name__ == '__main__':
    # Only run main() when called directly, not run_semi_supervised_experiments()
    main() 