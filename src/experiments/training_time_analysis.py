#!/usr/bin/env python3
"""
Training Time Analysis Experiment
Analyzes training time across different model configurations and epochs
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
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.baseline import MLP, SimpleCNN, ResNet
from src.models.noise_robust import RobustCNN, RobustResNet

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingTimeAnalyzer:
    """Analyze training time of different model configurations"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        
        # Create output directory
        self.output_dir = Path('experiments/training_time_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for analysis"""
        system_info = {
            'device': self.device,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            system_info['gpu_name'] = torch.cuda.get_device_name(0)
            system_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return system_info
    
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
    
    def train_model(self, model: nn.Module, dataloader: DataLoader, epochs: int) -> Dict[str, Any]:
        """Train model and measure timing"""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Initialize timing metrics
        timing_data = {
            'total_time': 0,
            'epoch_times': [],
            'batch_times': [],
            'forward_times': [],
            'backward_times': [],
            'optimizer_times': [],
            'memory_usage': []
        }
        
        # Get initial memory usage
        initial_memory = self.get_memory_usage()
        
        print(f"  Training for {epochs} epochs...")
        
        total_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_batch_times = []
            epoch_forward_times = []
            epoch_backward_times = []
            epoch_optimizer_times = []
            
            for batch_idx, (data, target) in enumerate(dataloader):
                batch_start_time = time.time()
                
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass timing
                forward_start = time.time()
                output = model(data)
                loss = criterion(output, target)
                forward_time = time.time() - forward_start
                epoch_forward_times.append(forward_time)
                
                # Backward pass timing
                backward_start = time.time()
                optimizer.zero_grad()
                loss.backward()
                backward_time = time.time() - backward_start
                epoch_backward_times.append(backward_time)
                
                # Optimizer step timing
                optimizer_start = time.time()
                optimizer.step()
                optimizer_time = time.time() - optimizer_start
                epoch_optimizer_times.append(optimizer_time)
                
                # Total batch time
                batch_time = time.time() - batch_start_time
                epoch_batch_times.append(batch_time)
                
                # Print progress every 10 batches
                if batch_idx % 10 == 0:
                    print(f"    Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, "
                          f"Loss: {loss.item():.4f}, Time: {batch_time:.3f}s")
            
            # Epoch timing
            epoch_time = time.time() - epoch_start_time
            timing_data['epoch_times'].append(epoch_time)
            timing_data['batch_times'].extend(epoch_batch_times)
            timing_data['forward_times'].extend(epoch_forward_times)
            timing_data['backward_times'].extend(epoch_backward_times)
            timing_data['optimizer_times'].extend(epoch_optimizer_times)
            
            # Memory usage at end of epoch
            memory_usage = self.get_memory_usage()
            timing_data['memory_usage'].append(memory_usage)
            
            print(f"  Epoch {epoch+1} completed in {epoch_time:.2f}s")
        
        # Total training time
        total_time = time.time() - total_start_time
        timing_data['total_time'] = total_time
        
        return timing_data
    
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
    
    def analyze_configuration(self, dataset: str, model_type: str, epochs: int) -> Dict[str, Any]:
        """Analyze training time for a specific configuration"""
        print(f"Analyzing {dataset}-{model_type}-{epochs}epochs...")
        
        try:
            # Create model and dataloader
            model = self.create_model(model_type, dataset)
            dataloader = self.get_dataloader(dataset)
            
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
            
            # Train model and measure timing
            timing_data = self.train_model(model, dataloader, epochs)
            
            # Calculate statistics
            timing_stats = {
                'total_time_minutes': timing_data['total_time'] / 60,
                'avg_epoch_time': sum(timing_data['epoch_times']) / len(timing_data['epoch_times']),
                'avg_batch_time': sum(timing_data['batch_times']) / len(timing_data['batch_times']),
                'avg_forward_time': sum(timing_data['forward_times']) / len(timing_data['forward_times']),
                'avg_backward_time': sum(timing_data['backward_times']) / len(timing_data['backward_times']),
                'avg_optimizer_time': sum(timing_data['optimizer_times']) / len(timing_data['optimizer_times']),
                'min_epoch_time': min(timing_data['epoch_times']),
                'max_epoch_time': max(timing_data['epoch_times']),
                'std_epoch_time': (sum((x - sum(timing_data['epoch_times'])/len(timing_data['epoch_times']))**2 
                                  for x in timing_data['epoch_times']) / len(timing_data['epoch_times']))**0.5
            }
            
            # Memory statistics
            memory_stats = {
                'initial_memory_gb': timing_data['memory_usage'][0]['rss_gb'],
                'max_memory_gb': max(m['rss_gb'] for m in timing_data['memory_usage']),
                'avg_memory_gb': sum(m['rss_gb'] for m in timing_data['memory_usage']) / len(timing_data['memory_usage']),
                'memory_increase_gb': max(m['rss_gb'] for m in timing_data['memory_usage']) - timing_data['memory_usage'][0]['rss_gb']
            }
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                'dataset': dataset,
                'model_type': model_type,
                'epochs': epochs,
                'device': self.device,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': model_size_mb,
                'timing_statistics': timing_stats,
                'memory_statistics': memory_stats,
                'detailed_timing': timing_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error analyzing {dataset}-{model_type}-{epochs}epochs: {e}")
            return {
                'dataset': dataset,
                'model_type': model_type,
                'epochs': epochs,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_analysis(self, datasets: List[str], model_types: List[str], 
                    epochs: int) -> Dict[str, Any]:
        """Run training time analysis for all configurations"""
        print("Starting training time analysis...")
        print(f"Datasets: {datasets}")
        print(f"Model types: {model_types}")
        print(f"Epochs: {epochs}")
        print(f"Device: {self.device}")
        print("-" * 60)
        
        # Get system information
        system_info = self.get_system_info()
        print(f"System Info: {system_info}")
        print("-" * 60)
        
        results = {
            'metadata': {
                'datasets': datasets,
                'model_types': model_types,
                'epochs': epochs,
                'device': self.device,
                'system_info': system_info,
                'timestamp': datetime.now().isoformat()
            },
            'configurations': []
        }
        
        total_configs = len(datasets) * len(model_types)
        current_config = 0
        
        for dataset in datasets:
            for model_type in model_types:
                current_config += 1
                print(f"\nProgress: {current_config}/{total_configs}")
                
                config_result = self.analyze_configuration(dataset, model_type, epochs)
                results['configurations'].append(config_result)
                
                # Save intermediate results
                if current_config % 3 == 0:
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
            'timing_statistics': {},
            'memory_statistics': {}
        }
        
        # Collect all timing data
        all_total_times = []
        all_epoch_times = []
        all_memory_usage = []
        
        # Group by dataset
        for config in valid_configs:
            dataset = config['dataset']
            if dataset not in summary['by_dataset']:
                summary['by_dataset'][dataset] = {
                    'count': 0,
                    'avg_total_time_minutes': 0,
                    'avg_epoch_time': 0,
                    'avg_memory_gb': 0,
                    'avg_parameters': 0
                }
            
            summary['by_dataset'][dataset]['count'] += 1
            
            if 'timing_statistics' in config:
                timing_stats = config['timing_statistics']
                all_total_times.append(timing_stats['total_time_minutes'])
                all_epoch_times.append(timing_stats['avg_epoch_time'])
                
                summary['by_dataset'][dataset]['avg_total_time_minutes'] += timing_stats['total_time_minutes']
                summary['by_dataset'][dataset]['avg_epoch_time'] += timing_stats['avg_epoch_time']
            
            if 'memory_statistics' in config:
                memory_stats = config['memory_statistics']
                all_memory_usage.append(memory_stats['avg_memory_gb'])
                summary['by_dataset'][dataset]['avg_memory_gb'] += memory_stats['avg_memory_gb']
            
            if 'total_parameters' in config:
                summary['by_dataset'][dataset]['avg_parameters'] += config['total_parameters']
        
        # Calculate averages for datasets
        for dataset in summary['by_dataset']:
            count = summary['by_dataset'][dataset]['count']
            summary['by_dataset'][dataset]['avg_total_time_minutes'] /= count
            summary['by_dataset'][dataset]['avg_epoch_time'] /= count
            summary['by_dataset'][dataset]['avg_memory_gb'] /= count
            summary['by_dataset'][dataset]['avg_parameters'] /= count
        
        # Group by model type
        for config in valid_configs:
            model_type = config['model_type']
            if model_type not in summary['by_model_type']:
                summary['by_model_type'][model_type] = {
                    'count': 0,
                    'avg_total_time_minutes': 0,
                    'avg_epoch_time': 0,
                    'avg_memory_gb': 0,
                    'avg_parameters': 0
                }
            
            summary['by_model_type'][model_type]['count'] += 1
            
            if 'timing_statistics' in config:
                timing_stats = config['timing_statistics']
                summary['by_model_type'][model_type]['avg_total_time_minutes'] += timing_stats['total_time_minutes']
                summary['by_model_type'][model_type]['avg_epoch_time'] += timing_stats['avg_epoch_time']
            
            if 'memory_statistics' in config:
                memory_stats = config['memory_statistics']
                summary['by_model_type'][model_type]['avg_memory_gb'] += memory_stats['avg_memory_gb']
            
            if 'total_parameters' in config:
                summary['by_model_type'][model_type]['avg_parameters'] += config['total_parameters']
        
        # Calculate averages for model types
        for model_type in summary['by_model_type']:
            count = summary['by_model_type'][model_type]['count']
            summary['by_model_type'][model_type]['avg_total_time_minutes'] /= count
            summary['by_model_type'][model_type]['avg_epoch_time'] /= count
            summary['by_model_type'][model_type]['avg_memory_gb'] /= count
            summary['by_model_type'][model_type]['avg_parameters'] /= count
        
        # Overall timing statistics
        if all_total_times:
            summary['timing_statistics'] = {
                'min_total_time_minutes': min(all_total_times),
                'max_total_time_minutes': max(all_total_times),
                'avg_total_time_minutes': sum(all_total_times) / len(all_total_times),
                'std_total_time_minutes': (sum((x - sum(all_total_times)/len(all_total_times))**2 
                                              for x in all_total_times) / len(all_total_times))**0.5,
                'min_epoch_time': min(all_epoch_times),
                'max_epoch_time': max(all_epoch_times),
                'avg_epoch_time': sum(all_epoch_times) / len(all_epoch_times)
            }
        
        # Overall memory statistics
        if all_memory_usage:
            summary['memory_statistics'] = {
                'min_memory_gb': min(all_memory_usage),
                'max_memory_gb': max(all_memory_usage),
                'avg_memory_gb': sum(all_memory_usage) / len(all_memory_usage)
            }
        
        return summary
    
    def save_results(self, results: Dict[str, Any]):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f'training_time_analysis_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {output_file}")
        return output_file
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("TRAINING TIME ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total configurations analyzed: {summary['total_configurations']}")
        print(f"Valid configurations: {summary['valid_configurations']}")
        print(f"Failed configurations: {summary['failed_configurations']}")
        
        if 'timing_statistics' in summary:
            timing_stats = summary['timing_statistics']
            print(f"\nTraining Time Statistics:")
            print(f"  Total Time Range: {timing_stats['min_total_time_minutes']:.2f} - {timing_stats['max_total_time_minutes']:.2f} minutes")
            print(f"  Average Total Time: {timing_stats['avg_total_time_minutes']:.2f} minutes")
            print(f"  Epoch Time Range: {timing_stats['min_epoch_time']:.2f} - {timing_stats['max_epoch_time']:.2f} seconds")
            print(f"  Average Epoch Time: {timing_stats['avg_epoch_time']:.2f} seconds")
        
        if 'memory_statistics' in summary:
            memory_stats = summary['memory_statistics']
            print(f"\nMemory Usage Statistics:")
            print(f"  Memory Range: {memory_stats['min_memory_gb']:.2f} - {memory_stats['max_memory_gb']:.2f} GB")
            print(f"  Average Memory: {memory_stats['avg_memory_gb']:.2f} GB")
        
        print("\nPerformance by Dataset:")
        for dataset, stats in summary['by_dataset'].items():
            print(f"  {dataset}: {stats['avg_total_time_minutes']:.2f} min avg, {stats['avg_epoch_time']:.2f}s/epoch")
        
        print("\nPerformance by Model Type:")
        for model_type, stats in summary['by_model_type'].items():
            print(f"  {model_type}: {stats['avg_total_time_minutes']:.2f} min avg, {stats['avg_parameters']:,.0f} params")

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Training Time Analysis Experiment')
    parser.add_argument('--datasets', nargs='+', default=['cifar10', 'mnist'],
                       help='Datasets to analyze')
    parser.add_argument('--model_types', nargs='+', 
                       default=['simple_cnn', 'robust_cnn', 'resnet', 'robust_resnet', 'mlp', 'robust_mlp'],
                       help='Model types to analyze')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs to train (default: 10)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Create analyzer
    analyzer = TrainingTimeAnalyzer(device=device)
    
    # Run analysis
    results = analyzer.run_analysis(args.datasets, args.model_types, args.epochs)
    
    # Save results
    output_file = analyzer.save_results(results)
    
    # Print summary
    analyzer.print_summary(results['summary'])
    
    print(f"\nTraining time analysis completed!")
    print(f"Results saved to: {output_file}")

if __name__ == '__main__':
    main() 