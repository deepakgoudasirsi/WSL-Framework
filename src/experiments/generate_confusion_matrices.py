#!/usr/bin/env python3
"""
Confusion Matrix Generator
Generates confusion matrices for model evaluation and analysis
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
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

from src.models.baseline import MLP, SimpleCNN, ResNet
from src.models.noise_robust import RobustCNN, RobustResNet

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfusionMatrixGenerator:
    """Generate confusion matrices for model evaluation"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.class_names = {
            'cifar10': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck'],
            'mnist': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        }
    
    def create_model(self, model_type: str, dataset: str, strategy: str = 'traditional') -> nn.Module:
        """Create model based on type and dataset"""
        if dataset == 'cifar10':
            num_classes = 10
            input_channels = 3
        elif dataset == 'mnist':
            num_classes = 10
            input_channels = 1
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        if model_type == 'mlp':
            # MLP needs different input sizes for different datasets
            if dataset == 'cifar10':
                input_size = 32 * 32 * 3  # CIFAR-10: 32x32x3
            elif dataset == 'mnist':
                input_size = 28 * 28 * 1  # MNIST: 28x28x1
            else:
                input_size = 784  # Default
            
            model = MLP(input_size=input_size, num_classes=num_classes)
        elif model_type == 'simple_cnn':
            # SimpleCNN is hardcoded for 3 channels (CIFAR-10), skip for MNIST
            if dataset == 'mnist':
                raise ValueError("SimpleCNN is not compatible with MNIST (1 channel). Use other models.")
            else:
                model = SimpleCNN(num_classes=num_classes)
        elif model_type == 'robust_cnn':
            model = RobustCNN(num_classes=num_classes, input_channels=input_channels, loss_type='gce')
        elif model_type == 'resnet':
            model = ResNet(num_classes=num_classes, in_channels=input_channels)
        elif model_type == 'robust_resnet':
            model = RobustResNet(num_classes=num_classes, loss_type='gce')
        elif model_type == 'robust_mlp':
            # Use regular MLP for robust_mlp since RobustMLP doesn't exist
            if dataset == 'cifar10':
                input_size = 32 * 32 * 3
            elif dataset == 'mnist':
                input_size = 28 * 28 * 1
            else:
                input_size = 784
            
            model = MLP(input_size=input_size, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model.to(self.device)
    
    def get_dataloader(self, dataset: str, batch_size: int = 128) -> DataLoader:
        """Get test dataloader for the specified dataset"""
        if dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            dataset_obj = datasets.CIFAR10('data', train=False, download=True, transform=transform)
        elif dataset == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            dataset_obj = datasets.MNIST('data', train=False, download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        return DataLoader(dataset_obj, batch_size=batch_size, shuffle=False, num_workers=0)
    
    def load_pretrained_model(self, model: nn.Module, experiment_dir: str) -> bool:
        """Load pretrained model weights if available"""
        model_path = Path(experiment_dir) / 'best_model.pth'
        
        if model_path.exists():
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded pretrained model from {model_path}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            print(f"No pretrained model found at {model_path}")
            return False
    
    def find_experiment_dir(self, dataset: str, model_type: str, strategy: str) -> Optional[str]:
        """Find experiment directory for the given configuration"""
        # Look in semi-supervised experiments
        if strategy in ['consistency', 'pseudo_label', 'co_training', 'mixmatch']:
            experiment_dir = Path('experiments/semi_supervised')
            pattern = f"{dataset}_{model_type}_labeled0.1_*"
            
            for exp_dir in experiment_dir.glob(pattern):
                config_file = exp_dir / 'config.json'
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    if config.get('strategy') == strategy:
                        return str(exp_dir)
        
        # Look in noise robustness experiments
        elif strategy == 'traditional':
            experiment_dir = Path('experiments/noise_robustness')
            pattern = f"noise_robustness_{dataset}_{model_type}_*"
            
            for exp_dir in experiment_dir.glob(pattern):
                config_file = exp_dir / 'config.json'
                if config_file.exists():
                    return str(exp_dir)
        
        # Look in baseline experiments
        elif strategy == 'combined':
            # For combined strategy, look for any available experiment
            experiment_dir = Path('experiments/semi_supervised')
            pattern = f"{dataset}_{model_type}_*"
            
            for exp_dir in experiment_dir.glob(pattern):
                config_file = exp_dir / 'config.json'
                if config_file.exists():
                    return str(exp_dir)
            
            # If not found in semi-supervised, look in noise robustness
            experiment_dir = Path('experiments/noise_robustness')
            pattern = f"noise_robustness_{dataset}_{model_type}_*"
            
            for exp_dir in experiment_dir.glob(pattern):
                config_file = exp_dir / 'config.json'
                if config_file.exists():
                    return str(exp_dir)
        
        return None
    
    def evaluate_model(self, model: nn.Module, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate model and get predictions"""
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                # Get probabilities
                probabilities = torch.softmax(output, dim=1)
                all_probabilities.append(probabilities.cpu().numpy())
                
                # Get predictions
                predictions = output.argmax(dim=1)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # Concatenate all batches
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        all_probabilities = np.concatenate(all_probabilities)
        
        return all_predictions, all_targets, all_probabilities
    
    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                class_names: List[str], normalize: bool = True) -> np.ndarray:
        """Generate confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            # Normalize by row (true labels)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)  # Replace NaN with 0
        
        return cm
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                            dataset: str, model_type: str, strategy: str,
                            output_file: str, normalize: bool = True):
        """Plot confusion matrix with professional styling"""
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        if normalize:
            fmt = '.2f'
            title = f'Normalized Confusion Matrix\n{dataset.upper()} - {model_type.replace("_", " ").title()} - {strategy.replace("_", " ").title()}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix\n{dataset.upper()} - {model_type.replace("_", " ").title()} - {strategy.replace("_", " ").title()}'
        
        # Create heatmap with custom styling
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Normalized Count' if normalize else 'Count'})
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save with high DPI
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {output_file}")
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     class_names: List[str]) -> str:
        """Generate detailed classification report"""
        report = classification_report(y_true, y_pred, target_names=class_names, 
                                     output_dict=False, digits=3)
        return report
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         probabilities: np.ndarray) -> Dict[str, float]:
        """Calculate various performance metrics"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        # Macro and weighted averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        # Weighted averages
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)
        
        # Top-1 and Top-5 accuracy
        top1_accuracy = accuracy
        
        # Calculate top-5 accuracy
        top5_indices = np.argsort(probabilities, axis=1)[:, -5:]
        top5_correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top5_indices[i]:
                top5_correct += 1
        top5_accuracy = top5_correct / len(y_true)
        
        return {
            'accuracy': accuracy,
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
            'support': support.tolist()
        }
    
    def generate_confusion_matrix_analysis(self, dataset: str, model_type: str, 
                                         strategy: str, output_file: str,
                                         use_pretrained: bool = True) -> Dict[str, Any]:
        """Generate comprehensive confusion matrix analysis"""
        print(f"Generating confusion matrix for {dataset}-{model_type}-{strategy}...")
        
        try:
            # Create model
            model = self.create_model(model_type, dataset, strategy)
            
            # Load pretrained weights if available and requested
            if use_pretrained:
                experiment_dir = self.find_experiment_dir(dataset, model_type, strategy)
                if experiment_dir:
                    self.load_pretrained_model(model, experiment_dir)
                else:
                    print("No pretrained model found. Using random weights.")
            
            # Get dataloader
            dataloader = self.get_dataloader(dataset)
            
            # Evaluate model
            print("Evaluating model...")
            predictions, targets, probabilities = self.evaluate_model(model, dataloader)
            
            # Get class names
            class_names = self.class_names[dataset]
            
            # Generate confusion matrix
            cm_normalized = self.generate_confusion_matrix(targets, predictions, class_names, normalize=True)
            cm_raw = self.generate_confusion_matrix(targets, predictions, class_names, normalize=False)
            
            # Calculate metrics
            metrics = self.calculate_metrics(targets, predictions, probabilities)
            
            # Generate classification report
            classification_rep = self.generate_classification_report(targets, predictions, class_names)
            
            # Plot confusion matrix
            self.plot_confusion_matrix(cm_normalized, class_names, dataset, model_type, strategy, 
                                     output_file, normalize=True)
            
            # Create results dictionary
            results = {
                'dataset': dataset,
                'model_type': model_type,
                'strategy': strategy,
                'metrics': metrics,
                'classification_report': classification_rep,
                'confusion_matrix_normalized': cm_normalized.tolist(),
                'confusion_matrix_raw': cm_raw.tolist(),
                'predictions': predictions.tolist(),
                'targets': targets.tolist(),
                'class_names': class_names,
                'timestamp': datetime.now().isoformat()
            }
            
            # Print summary
            print(f"\nConfusion Matrix Analysis Summary:")
            print(f"Dataset: {dataset}")
            print(f"Model: {model_type}")
            print(f"Strategy: {strategy}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
            print(f"Macro F1: {metrics['macro_f1']:.4f}")
            print(f"Output saved to: {output_file}")
            
            return results
            
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")
            return {'error': str(e)}
    
    def generate_multiple_confusion_matrices(self, configurations: List[Dict[str, str]], 
                                           output_dir: str = './confusion_matrices'):
        """Generate confusion matrices for multiple configurations"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, config in enumerate(configurations):
            print(f"\nProcessing configuration {i+1}/{len(configurations)}")
            
            dataset = config['dataset']
            model_type = config['model_type']
            strategy = config['strategy']
            
            output_file = output_path / f"{dataset}_{model_type}_{strategy}_confusion_matrix.png"
            
            result = self.generate_confusion_matrix_analysis(
                dataset, model_type, strategy, str(output_file)
            )
            
            results.append(result)
            
            # Save individual results
            result_file = output_path / f"{dataset}_{model_type}_{strategy}_results.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        
        # Save combined results
        combined_file = output_path / 'all_confusion_matrix_results.json'
        with open(combined_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nAll confusion matrices generated in {output_dir}")
        return results

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Generate Confusion Matrices')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'mnist'],
                       help='Dataset to analyze')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['simple_cnn', 'robust_cnn', 'resnet', 'robust_resnet', 'mlp', 'robust_mlp'],
                       help='Model type to analyze')
    parser.add_argument('--strategy', type=str, default='combined',
                       choices=['traditional', 'consistency', 'pseudo_label', 'co_training', 'mixmatch', 'combined'],
                       help='Training strategy used')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output file path for confusion matrix')
    parser.add_argument('--no_pretrained', action='store_true',
                       help='Use random weights instead of pretrained model')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Create generator
    generator = ConfusionMatrixGenerator()
    
    # Generate confusion matrix
    results = generator.generate_confusion_matrix_analysis(
        args.dataset, args.model_type, args.strategy, args.output_file,
        use_pretrained=not args.no_pretrained
    )
    
    if 'error' not in results:
        print(f"\nConfusion matrix generation completed successfully!")
        print(f"Results saved to: {args.output_file}")
    else:
        print(f"\nError: {results['error']}")

if __name__ == '__main__':
    main() 