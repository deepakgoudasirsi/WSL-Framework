#!/usr/bin/env python3
"""
Memory Analysis Experiment
Analyzes memory usage across different model configurations and batch sizes
"""

# Import warning suppression utility first
from src.utils.warning_suppression import apply_warning_suppression
apply_warning_suppression()

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.baseline import MLP, SimpleCNN, ResNet
from src.models.noise_robust import RobustCNN, RobustResNet

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryAnalyzer:
    """Analyze memory usage of different model configurations"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        
        # Create output directory
        self.output_dir = Path('experiments/memory_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        memory_usage = {
            'rss_gb': memory_info.rss / (1024**3),  # Resident Set Size
            'vms_gb': memory_info.vms / (1024**3),  # Virtual Memory Size
        }
        
        # Get GPU memory if available
        if torch.cuda.is_available():
            memory_usage['gpu_gb'] = torch.cuda.memory_allocated() / (1024**3)
            memory_usage['gpu_cached_gb'] = torch.cuda.memory_reserved() / (1024**3)
        
        return memory_usage
    
    def create_model(self, model_type: str, dataset: str) -> nn.Module:
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
            model = ResNet(num_classes=num_classes)
        elif model_type == 'robust_resnet':
            model = RobustResNet(num_classes=num_classes, loss_type='gce')
        elif model_type == 'robust_mlp':
            # Use regular MLP for robust_mlp since RobustMLP doesn't exist
            model = MLP(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model.to(self.device)
    
    def get_dataloader(self, dataset: str, batch_size: int) -> DataLoader:
        """Get dataloader for the specified dataset"""
        if dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            dataset_obj = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        elif dataset == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            dataset_obj = datasets.MNIST('data', train=True, download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        return DataLoader(dataset_obj, batch_size=batch_size, shuffle=True, num_workers=0)
    
    def measure_model_memory(self, model: nn.Module, dataloader: DataLoader, 
                           num_batches: int = 5) -> Dict[str, float]:
        """Measure memory usage during model training"""
        model.train()
        
        # Get initial memory usage
        initial_memory = self.get_memory_usage()
        
        # Run training for a few batches
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        max_memory = initial_memory.copy()
        memory_samples = []
        
        for i, (data, target) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            data, target = data.to(self.device), target.to(self.device)
            
            # Measure memory before forward pass
            memory_before = self.get_memory_usage()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Measure memory after forward pass
            memory_after_forward = self.get_memory_usage()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Measure memory after backward pass
            memory_after_backward = self.get_memory_usage()
            
            # Update weights
            optimizer.step()
            
            # Measure memory after optimizer step
            memory_after_step = self.get_memory_usage()
            
            # Store memory samples
            memory_samples.append({
                'before': memory_before,
                'after_forward': memory_after_forward,
                'after_backward': memory_after_backward,
                'after_step': memory_after_step
            })
            
            # Update max memory usage
            for key in max_memory:
                max_memory[key] = max(max_memory[key], memory_after_step[key])
        
        # Calculate average memory usage
        avg_memory = {}
        for key in initial_memory:
            values = [sample['after_step'][key] for sample in memory_samples]
            avg_memory[key] = sum(values) / len(values)
        
        # Calculate memory increase
        memory_increase = {}
        for key in initial_memory:
            memory_increase[key] = max_memory[key] - initial_memory[key]
        
        return {
            'initial_memory': initial_memory,
            'max_memory': max_memory,
            'avg_memory': avg_memory,
            'memory_increase': memory_increase,
            'memory_samples': memory_samples
        }
    
    def analyze_configuration(self, dataset: str, model_type: str, batch_size: int) -> Dict[str, Any]:
        """Analyze memory usage for a specific configuration"""
        print(f"Analyzing {dataset}-{model_type}-batch{batch_size}...")
        
        try:
            # Create model and dataloader
            model = self.create_model(model_type, dataset)
            dataloader = self.get_dataloader(dataset, batch_size)
            
            # Measure memory usage
            memory_results = self.measure_model_memory(model, dataloader)
            
            # Count model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Get model size in MB
            param_size = 0
            buffer_size = 0
            
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            model_size_mb = (param_size + buffer_size) / (1024**2)
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                'dataset': dataset,
                'model_type': model_type,
                'batch_size': batch_size,
                'device': self.device,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': model_size_mb,
                'memory_usage': memory_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error analyzing {dataset}-{model_type}-batch{batch_size}: {e}")
            return {
                'dataset': dataset,
                'model_type': model_type,
                'batch_size': batch_size,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_analysis(self, datasets: List[str], model_types: List[str], 
                    batch_sizes: List[int]) -> Dict[str, Any]:
        """Run memory analysis for all configurations"""
        print("Starting memory analysis...")
        print(f"Datasets: {datasets}")
        print(f"Model types: {model_types}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Device: {self.device}")
        print("-" * 60)
        
        results = {
            'metadata': {
                'datasets': datasets,
                'model_types': model_types,
                'batch_sizes': batch_sizes,
                'device': self.device,
                'timestamp': datetime.now().isoformat()
            },
            'configurations': []
        }
        
        total_configs = len(datasets) * len(model_types) * len(batch_sizes)
        current_config = 0
        
        for dataset in datasets:
            for model_type in model_types:
                for batch_size in batch_sizes:
                    current_config += 1
                    print(f"\nProgress: {current_config}/{total_configs}")
                    
                    config_result = self.analyze_configuration(dataset, model_type, batch_size)
                    results['configurations'].append(config_result)
                    
                    # Save intermediate results
                    if current_config % 5 == 0:
                        self.save_results(results)
                        print(f"Intermediate results saved ({current_config}/{total_configs})")
        
        # Generate summary statistics
        results['summary'] = self.generate_summary(results['configurations'])
        
        return results
    
    def generate_summary(self, configurations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from analysis results"""
        print("Generating summary statistics...")
        
        # Filter out configurations with errors
        valid_configs = [config for config in configurations if 'error' not in config]
        
        if not valid_configs:
            return {'error': 'No valid configurations found'}
        
        summary = {
            'total_configurations': len(configurations),
            'valid_configurations': len(valid_configs),
            'failed_configurations': len(configurations) - len(valid_configs),
            'by_dataset': {},
            'by_model_type': {},
            'by_batch_size': {},
            'memory_statistics': {}
        }
        
        # Group by dataset
        for config in valid_configs:
            dataset = config['dataset']
            if dataset not in summary['by_dataset']:
                summary['by_dataset'][dataset] = {
                    'count': 0,
                    'avg_memory_gb': 0,
                    'max_memory_gb': 0,
                    'avg_model_size_mb': 0
                }
            
            summary['by_dataset'][dataset]['count'] += 1
            
            if 'memory_usage' in config:
                memory_gb = config['memory_usage']['max_memory'].get('rss_gb', 0)
                summary['by_dataset'][dataset]['avg_memory_gb'] += memory_gb
                summary['by_dataset'][dataset]['max_memory_gb'] = max(
                    summary['by_dataset'][dataset]['max_memory_gb'], memory_gb
                )
            
            if 'model_size_mb' in config:
                summary['by_dataset'][dataset]['avg_model_size_mb'] += config['model_size_mb']
        
        # Calculate averages
        for dataset in summary['by_dataset']:
            count = summary['by_dataset'][dataset]['count']
            summary['by_dataset'][dataset]['avg_memory_gb'] /= count
            summary['by_dataset'][dataset]['avg_model_size_mb'] /= count
        
        # Group by model type
        for config in valid_configs:
            model_type = config['model_type']
            if model_type not in summary['by_model_type']:
                summary['by_model_type'][model_type] = {
                    'count': 0,
                    'avg_memory_gb': 0,
                    'max_memory_gb': 0,
                    'avg_parameters': 0
                }
            
            summary['by_model_type'][model_type]['count'] += 1
            
            if 'memory_usage' in config:
                memory_gb = config['memory_usage']['max_memory'].get('rss_gb', 0)
                summary['by_model_type'][model_type]['avg_memory_gb'] += memory_gb
                summary['by_model_type'][model_type]['max_memory_gb'] = max(
                    summary['by_model_type'][model_type]['max_memory_gb'], memory_gb
                )
            
            if 'total_parameters' in config:
                summary['by_model_type'][model_type]['avg_parameters'] += config['total_parameters']
        
        # Calculate averages
        for model_type in summary['by_model_type']:
            count = summary['by_model_type'][model_type]['count']
            summary['by_model_type'][model_type]['avg_memory_gb'] /= count
            summary['by_model_type'][model_type]['avg_parameters'] /= count
        
        # Group by batch size
        for config in valid_configs:
            batch_size = config['batch_size']
            if batch_size not in summary['by_batch_size']:
                summary['by_batch_size'][batch_size] = {
                    'count': 0,
                    'avg_memory_gb': 0,
                    'max_memory_gb': 0
                }
            
            summary['by_batch_size'][batch_size]['count'] += 1
            
            if 'memory_usage' in config:
                memory_gb = config['memory_usage']['max_memory'].get('rss_gb', 0)
                summary['by_batch_size'][batch_size]['avg_memory_gb'] += memory_gb
                summary['by_batch_size'][batch_size]['max_memory_gb'] = max(
                    summary['by_batch_size'][batch_size]['max_memory_gb'], memory_gb
                )
        
        # Calculate averages
        for batch_size in summary['by_batch_size']:
            count = summary['by_batch_size'][batch_size]['count']
            summary['by_batch_size'][batch_size]['avg_memory_gb'] /= count
        
        # Overall memory statistics
        all_memory_gb = []
        all_model_sizes_mb = []
        all_parameters = []
        
        for config in valid_configs:
            if 'memory_usage' in config:
                all_memory_gb.append(config['memory_usage']['max_memory'].get('rss_gb', 0))
            if 'model_size_mb' in config:
                all_model_sizes_mb.append(config['model_size_mb'])
            if 'total_parameters' in config:
                all_parameters.append(config['total_parameters'])
        
        if all_memory_gb:
            summary['memory_statistics'] = {
                'min_memory_gb': min(all_memory_gb),
                'max_memory_gb': max(all_memory_gb),
                'avg_memory_gb': sum(all_memory_gb) / len(all_memory_gb),
                'std_memory_gb': (sum((x - sum(all_memory_gb)/len(all_memory_gb))**2 for x in all_memory_gb) / len(all_memory_gb))**0.5
            }
        
        if all_model_sizes_mb:
            summary['model_size_statistics'] = {
                'min_model_size_mb': min(all_model_sizes_mb),
                'max_model_size_mb': max(all_model_sizes_mb),
                'avg_model_size_mb': sum(all_model_sizes_mb) / len(all_model_sizes_mb)
            }
        
        if all_parameters:
            summary['parameter_statistics'] = {
                'min_parameters': min(all_parameters),
                'max_parameters': max(all_parameters),
                'avg_parameters': sum(all_parameters) / len(all_parameters)
            }
        
        return summary
    
    def save_results(self, results: Dict[str, Any]):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f'memory_analysis_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {output_file}")
        return output_file
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("MEMORY ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total configurations analyzed: {summary['total_configurations']}")
        print(f"Valid configurations: {summary['valid_configurations']}")
        print(f"Failed configurations: {summary['failed_configurations']}")
        
        if 'memory_statistics' in summary:
            mem_stats = summary['memory_statistics']
            print(f"\nMemory Usage Statistics:")
            print(f"  Min: {mem_stats['min_memory_gb']:.2f} GB")
            print(f"  Max: {mem_stats['max_memory_gb']:.2f} GB")
            print(f"  Average: {mem_stats['avg_memory_gb']:.2f} GB")
            print(f"  Std Dev: {mem_stats['std_memory_gb']:.2f} GB")
        
        if 'model_size_statistics' in summary:
            size_stats = summary['model_size_statistics']
            print(f"\nModel Size Statistics:")
            print(f"  Min: {size_stats['min_model_size_mb']:.2f} MB")
            print(f"  Max: {size_stats['max_model_size_mb']:.2f} MB")
            print(f"  Average: {size_stats['avg_model_size_mb']:.2f} MB")
        
        if 'parameter_statistics' in summary:
            param_stats = summary['parameter_statistics']
            print(f"\nParameter Statistics:")
            print(f"  Min: {param_stats['min_parameters']:,}")
            print(f"  Max: {param_stats['max_parameters']:,}")
            print(f"  Average: {param_stats['avg_parameters']:,.0f}")
        
        print("\nPerformance by Dataset:")
        for dataset, stats in summary['by_dataset'].items():
            print(f"  {dataset}: {stats['avg_memory_gb']:.2f} GB avg, {stats['max_memory_gb']:.2f} GB max")
        
        print("\nPerformance by Model Type:")
        for model_type, stats in summary['by_model_type'].items():
            print(f"  {model_type}: {stats['avg_memory_gb']:.2f} GB avg, {stats['avg_parameters']:,.0f} params")
        
        print("\nPerformance by Batch Size:")
        for batch_size, stats in summary['by_batch_size'].items():
            print(f"  Batch {batch_size}: {stats['avg_memory_gb']:.2f} GB avg, {stats['max_memory_gb']:.2f} GB max")

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Memory Analysis Experiment')
    parser.add_argument('--datasets', nargs='+', default=['cifar10', 'mnist'],
                       help='Datasets to analyze')
    parser.add_argument('--model_types', nargs='+', 
                       default=['simple_cnn', 'robust_cnn', 'resnet', 'robust_resnet', 'mlp', 'robust_mlp'],
                       help='Model types to analyze')
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[32, 64, 128, 256],
                       help='Batch sizes to analyze')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Create analyzer
    analyzer = MemoryAnalyzer(device=device)
    
    # Run analysis
    results = analyzer.run_analysis(args.datasets, args.model_types, args.batch_sizes)
    
    # Save results
    output_file = analyzer.save_results(results)
    
    # Print summary
    analyzer.print_summary(results['summary'])
    
    print(f"\nMemory analysis completed!")
    print(f"Results saved to: {output_file}")

if __name__ == '__main__':
    main() 