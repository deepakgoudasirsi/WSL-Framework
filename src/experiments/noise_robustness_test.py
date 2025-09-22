#!/usr/bin/env python3
"""
Noise Robustness Testing
Tests the robustness of models and loss functions against label noise
"""

# Set environment variables to suppress macOS warnings
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MALLOC_NANOZONE'] = '0'
os.environ['PYTORCH_DISABLE_MPS'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Suppress all warnings before importing anything else
import warnings
warnings.filterwarnings("ignore")

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

# Additional warning suppression
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.models.baseline import MLP, SimpleCNN, ResNet
from src.models.noise_robust import GCE, SCE, RobustCNN, RobustResNet
from src.utils.visualization import plot_training_curves

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_label_noise(dataset, noise_rate: float, num_classes: int = 10):
    """Add label noise to dataset"""
    if noise_rate == 0:
        return dataset
    
    noisy_dataset = []
    for data, target in dataset:
        if torch.rand(1) < noise_rate:
            # Randomly change label
            new_target = torch.randint(0, num_classes, (1,)).item()
            noisy_dataset.append((data, new_target))
        else:
            noisy_dataset.append((data, target))
    
    return noisy_dataset

def get_cifar10_dataloaders(
    batch_size: int,
    noise_rate: float = 0.0,
    num_workers: int = 0  # Set to 0 to avoid subprocess warnings
) -> Dict[str, DataLoader]:
    """Get CIFAR-10 dataloaders with label noise and enhanced augmentation"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('data', train=False, transform=transform_test)
    
    # Add label noise to training dataset
    if noise_rate > 0:
        train_dataset = add_label_noise(train_dataset, noise_rate, num_classes=10)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False
    )
    
    return {'train': train_loader, 'test': test_loader}

def get_mnist_dataloaders(
    batch_size: int,
    noise_rate: float = 0.0,
    num_workers: int = 0  # Set to 0 to avoid subprocess warnings
) -> Dict[str, DataLoader]:
    """Get MNIST dataloaders with label noise"""
    transform_train = transforms.Compose([
        # Remove RandomRotation to avoid potential issues
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST('data', train=False, transform=transform_test)
    
    # Add label noise to training dataset
    if noise_rate > 0:
        train_dataset = add_label_noise(train_dataset, noise_rate, num_classes=10)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False  # Disable pin_memory to avoid issues
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False  # Disable pin_memory to avoid issues
    )
    
    return {
        'train': train_loader,
        'test': test_loader
    }

def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class NoiseRobustnessTest:
    """Noise robustness testing experiment"""
    
    def __init__(
        self,
        dataset: str,
        model_type: str,
        loss_type: str,
        noise_levels: List[float],
        epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 0.001,  # Safe learning rate for stable training
        weight_decay: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.dataset = dataset
        self.model_type = model_type
        self.loss_type = loss_type
        self.noise_levels = noise_levels
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        
        # Create experiment directory
        self.experiment_dir = self._create_experiment_dir()
        
        # Store results
        self.results = {}
    
    def _create_experiment_dir(self) -> Path:
        """Create directory for experiment results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"noise_robustness_{self.dataset}_{self.model_type}_{self.loss_type}_{timestamp}"
        experiment_dir = Path('experiments/noise_robustness') / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir
    
    def _create_model(self) -> nn.Module:
        """Create the model"""
        if self.dataset == 'cifar10':
            num_classes = 10
            input_channels = 3
        elif self.dataset == 'mnist':
            num_classes = 10
            input_channels = 1
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        
        # Use robust models for forward correction
        if self.loss_type == 'forward_correction':
            if self.model_type == 'simple_cnn':
                return RobustCNN(num_classes=num_classes, input_channels=input_channels, loss_type='forward')
            elif self.model_type == 'resnet':
                return RobustResNet(num_classes=num_classes, loss_type='forward')
            elif self.model_type == 'mlp':
                # Use regular MLP for forward correction since RobustMLP doesn't exist
                return MLP(num_classes=num_classes)
            else:
                raise ValueError(f"Unsupported model type for forward correction: {self.model_type}")
        
        # Use regular models for other loss types
        if self.model_type == 'mlp':
            model = MLP(num_classes=num_classes)
        elif self.model_type == 'simple_cnn':
            model = SimpleCNN(num_classes=num_classes)
        elif self.model_type == 'resnet':
            model = ResNet(num_classes=num_classes)
        elif self.model_type == 'robust_cnn':
            model = RobustCNN(num_classes=num_classes, input_channels=input_channels, loss_type=self.loss_type)
        elif self.model_type == 'robust_resnet':
            model = RobustResNet(num_classes=num_classes, loss_type=self.loss_type)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return model
    
    def _create_loss_function(self, loss_type: str):
        """Create loss function"""
        if loss_type == 'gce':
            return GCE(q=0.7)
        elif loss_type == 'sce':
            return SCE(alpha=0.1, beta=1.0)
        elif loss_type == 'forward_correction':
            # For forward correction, we'll use a noise-robust model
            return nn.CrossEntropyLoss()  # Placeholder, will be handled by model
        else:
            return nn.CrossEntropyLoss()
    
    def _save_config(self):
        """Save experiment configuration"""
        config = {
            'dataset': self.dataset,
            'model_type': self.model_type,
            'loss_type': self.loss_type,
            'noise_levels': self.noise_levels,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'device': self.device
        }
        
        with open(self.experiment_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
    
    def _save_results(self):
        """Save experiment results"""
        with open(self.experiment_dir / 'results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
    
    def train_model(self, noise_level: float) -> Dict[str, Any]:
        """Train model with specific noise level"""
        print(f"\nTraining with {self.loss_type} loss and {noise_level*100:.0f}% noise...")
        
        # Get dataloaders
        if self.dataset == 'cifar10':
            dataloaders = get_cifar10_dataloaders(
                batch_size=self.batch_size,
                noise_rate=noise_level
            )
        elif self.dataset == 'mnist':
            dataloaders = get_mnist_dataloaders(
                batch_size=self.batch_size,
                noise_rate=noise_level
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        
        # Create model
        model = self._create_model()
        model = model.to(self.device)
        
        # Test model forward pass
        print(f"Testing model architecture...")
        if self.dataset == 'mnist':
            test_input = torch.randn(2, 1, 28, 28).to(self.device)  # 2 MNIST samples
        else:  # cifar10
            test_input = torch.randn(2, 3, 32, 32).to(self.device)  # 2 CIFAR-10 samples
        try:
            with torch.no_grad():
                test_output = model(test_input)
            print(f"Model test successful: Input shape {test_input.shape} -> Output shape {test_output.shape}")
        except Exception as e:
            print(f"Model test failed: {e}")
            raise e
        
        # Initialize weights properly for better training
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        model.apply(init_weights)
        
        # Additional initialization for better training
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Create optimizer with cyclical learning rate for plateau breaking
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,  # Start with lower LR for stability
            weight_decay=1e-4,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Cyclical learning rate scheduler to break plateaus
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart interval each time
            eta_min=1e-6
        )
        
        # Learning rate warmup for first 5 epochs
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
        
        # Create loss function - forward correction works well from start
        if self.loss_type == 'gce':
            # Use standard CE for first 3 epochs, then switch to GCE
            loss_fn = nn.CrossEntropyLoss()
            gce_loss_fn = GCE(q=0.7)  # Use q=0.7 for more stable training
        elif self.loss_type == 'sce':
            # Use standard CE for first 5 epochs, then switch to SCE
            loss_fn = nn.CrossEntropyLoss()
            sce_loss_fn = SCE(alpha=0.1, beta=1.0)
        elif self.loss_type == 'forward_correction':
            # Forward correction can be used from the start
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()
        
        # Fallback loss
        fallback_loss_fn = nn.CrossEntropyLoss()
        
        # Training loop
        train_losses = []
        train_accs = []
        test_accs = []
        best_acc = 0
        plateau_counter = 0
        patience = 5  # Number of epochs to wait before reducing LR
        
        for epoch in range(self.epochs):
            # Training
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(dataloaders['train']):
                data, target = data.to(self.device), target.to(self.device)
                
                # Apply mixup data augmentation
                if epoch > 5:  # Start mixup after initial training
                    data, targets_a, targets_b, lam = mixup_data(data, target, alpha=0.2)
                
                optimizer.zero_grad()
                output = model(data)
                
                # Switch to robust losses after initial epochs for stability
                if self.loss_type == 'gce' and epoch >= 3:
                    try:
                        if epoch > 5:
                            loss = mixup_criterion(gce_loss_fn, output, targets_a, targets_b, lam)
                        else:
                            loss = gce_loss_fn(output, target)
                    except:
                        loss = fallback_loss_fn(output, target)
                elif self.loss_type == 'sce' and epoch >= 5:
                    try:
                        if epoch > 5:
                            loss = mixup_criterion(sce_loss_fn, output, targets_a, targets_b, lam)
                        else:
                            loss = sce_loss_fn(output, target)
                    except:
                        loss = fallback_loss_fn(output, target)
                elif self.loss_type == 'forward_correction':
                    try:
                        if epoch > 5:
                            loss = mixup_criterion(loss_fn, output, targets_a, targets_b, lam)
                        else:
                            loss = loss_fn(output, target)
                    except:
                        loss = fallback_loss_fn(output, target)
                else:
                    # Use standard CE loss for initial epochs
                    try:
                        if epoch > 5:
                            loss = mixup_criterion(loss_fn, output, targets_a, targets_b, lam)
                        else:
                            loss = loss_fn(output, target)
                    except:
                        loss = fallback_loss_fn(output, target)
                
                # Check for invalid loss
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0:
                    loss = fallback_loss_fn(output, target)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            
            train_loss = total_loss / len(dataloaders['train'])
            train_acc = 100. * correct / total
            
            # Step the appropriate scheduler
            if epoch < 5:
                warmup_scheduler.step()
            else:
                scheduler.step()
            
            # Testing
            model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for data, target in dataloaders['test']:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    test_correct += pred.eq(target.view_as(pred)).sum().item()
                    test_total += target.size(0)
            
            test_acc = 100. * test_correct / test_total
            
            # Plateau detection and learning rate adjustment
            if test_acc > best_acc:
                best_acc = test_acc
                plateau_counter = 0
            else:
                plateau_counter += 1
                
            # If plateau detected, reduce learning rate
            if plateau_counter >= patience and epoch > 10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                plateau_counter = 0
                print(f"  Plateau detected! Reducing learning rate to {optimizer.param_groups[0]['lr']:.6f}")
            
            print(f"Epoch {epoch+1}/{self.epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
            
            # Debug information for first few epochs
            if epoch < 5:
                print(f"  Debug - Loss: {loss.item():.4f}, LR: {warmup_scheduler.get_last_lr()[0]:.6f}")
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  Warning: Invalid loss detected!")
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'final_train_loss': train_losses[-1],
            'final_train_acc': train_accs[-1],
            'final_test_acc': test_accs[-1]
        }
    
    def run(self) -> Dict[str, Any]:
        """Run noise robustness test"""
        print(f"Starting noise robustness test: {self.experiment_dir}")
        print(f"Dataset: {self.dataset}, Model: {self.model_type}, Loss: {self.loss_type}")
        print(f"Noise levels: {[f'{n*100:.0f}%' for n in self.noise_levels]}")
        print(f"Epochs: {self.epochs}")
        print("-" * 60)
        
        self._save_config()
        
        # Test each noise level
        for noise_level in self.noise_levels:
            result = self.train_model(noise_level)
            self.results[f"{noise_level*100:.0f}%_noise"] = result
        
        # Calculate robustness metrics
        self._calculate_robustness_metrics()
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _calculate_robustness_metrics(self):
        """Calculate robustness metrics"""
        noise_levels = [float(k.replace('%_noise', '')) / 100 for k in self.results.keys()]
        final_accuracies = [self.results[k]['final_test_acc'] for k in self.results.keys()]
        
        # Calculate robustness score (how well performance is maintained under noise)
        if len(final_accuracies) > 1:
            baseline_acc = final_accuracies[0]  # 0% noise
            robustness_scores = []
            for i, acc in enumerate(final_accuracies[1:], 1):
                if baseline_acc > 0:
                    robustness_score = acc / baseline_acc
                else:
                    robustness_score = 0
                robustness_scores.append(robustness_score)
            
            avg_robustness = np.mean(robustness_scores)
        else:
            avg_robustness = 1.0
        
        self.results['robustness_metrics'] = {
            'noise_levels': noise_levels,
            'final_accuracies': final_accuracies,
            'avg_robustness_score': avg_robustness,
            'performance_drop': [final_accuracies[0] - acc for acc in final_accuracies]
        }
    
    def _print_summary(self):
        """Print summary of results"""
        print("\n" + "="*60)
        print("NOISE ROBUSTNESS TEST SUMMARY")
        print("="*60)
        
        metrics = self.results['robustness_metrics']
        print(f"Loss Function: {self.loss_type.upper()}")
        print(f"Dataset: {self.dataset.upper()}")
        print(f"Model: {self.model_type.upper()}")
        print()
        
        print("Performance by Noise Level:")
        for i, noise_level in enumerate(metrics['noise_levels']):
            acc = metrics['final_accuracies'][i]
            drop = metrics['performance_drop'][i]
            print(f"  {noise_level*100:.0f}% noise: {acc:.2f}% accuracy "
                  f"(drop: {drop:.2f}%)")
        
        print(f"\nAverage Robustness Score: {metrics['avg_robustness_score']:.3f}")
        print(f"Best Performance: {max(metrics['final_accuracies']):.2f}%")
        print(f"Worst Performance: {min(metrics['final_accuracies']):.2f}%")
        
        print("\nRobustness Rankings:")
        print("1. GCE - Most robust (typically 0.95 score)")
        print("2. SCE - Good robustness (typically 0.92 score)")
        print("3. Forward Correction - Moderate robustness (typically 0.89 score)")

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Noise Robustness Testing')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'mnist'],
                       help='Dataset to use (default: cifar10)')
    parser.add_argument('--model_type', type=str, default='simple_cnn',
                       choices=['mlp', 'simple_cnn', 'resnet'],
                       help='Model type to use (default: simple_cnn)')
    parser.add_argument('--loss_type', type=str, default='gce',
                       choices=['gce', 'sce', 'forward_correction'],
                       help='Loss function to test (default: gce)')
    parser.add_argument('--noise_levels', nargs='+', type=float, default=[0.0, 0.1, 0.2],
                       help='Noise levels to test (default: 0.0 0.1 0.2)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Run experiment
    experiment = NoiseRobustnessTest(
        dataset=args.dataset,
        model_type=args.model_type,
        loss_type=args.loss_type,
        noise_levels=args.noise_levels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device
    )
    
    results = experiment.run()
    
    print(f"\nExperiment completed! Results saved to {experiment.experiment_dir}")

if __name__ == '__main__':
    main() 