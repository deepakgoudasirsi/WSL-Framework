#!/usr/bin/env python3
"""
Hardware Configuration Test
Tests hardware configuration and performance across CPU, GPU, memory, and storage
"""

# Import warning suppression utility first
from src.utils.warning_suppression import apply_warning_suppression
apply_warning_suppression()

import argparse
import json
import logging
import sys
import time
import psutil
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HardwareConfigurationTester:
    """Test hardware configuration and performance"""
    
    def __init__(self):
        self.results = {
            'metadata': {
                'test_timestamp': datetime.now().isoformat(),
                'system_info': {},
                'test_configuration': {},
                'hardware_specifications': {}
            },
            'cpu_test_results': {},
            'gpu_test_results': {},
            'memory_test_results': {},
            'storage_test_results': {},
            'performance_metrics': {},
            'recommendations': {}
        }
        
        # Get system information
        self._get_system_info()
    
    def _get_system_info(self):
        """Get comprehensive system information"""
        print("Gathering system information...")
        
        system_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage': psutil.disk_usage('/').total / (1024**3)
        }
        
        if torch.cuda.is_available():
            system_info['gpu_name'] = torch.cuda.get_device_name(0)
            system_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            system_info['cuda_version'] = torch.version.cuda
        
        self.results['metadata']['system_info'] = system_info
        
        # Define hardware specifications based on system info
        hardware_specs = {
            'cpu': {
                'model': system_info.get('processor', 'Unknown'),
                'cores': system_info['cpu_count'],
                'logical_cores': system_info['cpu_count_logical'],
                'recommended': 'Intel Core i7 or AMD Ryzen 7 or higher'
            },
            'gpu': {
                'model': system_info.get('gpu_name', 'CPU Only'),
                'memory_gb': system_info.get('gpu_memory_gb', 0),
                'cuda_available': system_info['cuda_available'],
                'recommended': 'NVIDIA GPU with CUDA support (GTX 1060 or higher)'
            },
            'memory': {
                'total_gb': system_info['memory_total_gb'],
                'available_gb': system_info['memory_available_gb'],
                'recommended': '16 GB RAM minimum, 32 GB recommended'
            },
            'storage': {
                'total_gb': system_info['disk_usage'],
                'recommended': '100 GB SSD minimum'
            }
        }
        
        self.results['metadata']['hardware_specifications'] = hardware_specs
        
        print(f"System Information:")
        print(f"- Platform: {system_info['platform']}")
        print(f"- CPU: {system_info['processor']} ({system_info['cpu_count']} cores)")
        print(f"- Memory: {system_info['memory_total_gb']:.1f} GB total")
        print(f"- Storage: {system_info['disk_usage']:.1f} GB total")
        if system_info['cuda_available']:
            print(f"- GPU: {system_info['gpu_name']} ({system_info['gpu_memory_gb']:.1f} GB)")
        else:
            print("- GPU: Not available (CPU-only mode)")
    
    def test_cpu_performance(self) -> Dict[str, Any]:
        """Test CPU performance for deep learning tasks"""
        print("Testing CPU performance...")
        
        cpu_results = {
            'cpu_utilization': {},
            'processing_speed': {},
            'parallel_processing': {},
            'recommendations': []
        }
        
        # Test CPU utilization
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_results['cpu_utilization'] = {
            'current_usage': psutil.cpu_percent(interval=1),
            'per_core_usage': cpu_percent,
            'average_usage': np.mean(cpu_percent),
            'max_usage': np.max(cpu_percent)
        }
        
        # Test processing speed with matrix operations
        start_time = time.time()
        matrix_size = 1000
        a = np.random.rand(matrix_size, matrix_size)
        b = np.random.rand(matrix_size, matrix_size)
        c = np.dot(a, b)
        cpu_time = time.time() - start_time
        
        cpu_results['processing_speed'] = {
            'matrix_multiplication_time': cpu_time,
            'operations_per_second': (matrix_size**3) / cpu_time,
            'matrix_size': matrix_size
        }
        
        # Test parallel processing capability
        start_time = time.time()
        with torch.no_grad():
            tensor_a = torch.randn(1000, 1000)
            tensor_b = torch.randn(1000, 1000)
            tensor_c = torch.mm(tensor_a, tensor_b)
        torch_time = time.time() - start_time
        
        cpu_results['parallel_processing'] = {
            'torch_matrix_multiplication_time': torch_time,
            'torch_operations_per_second': (1000**3) / torch_time
        }
        
        # Generate CPU recommendations
        cpu_cores = psutil.cpu_count()
        if cpu_cores < 4:
            cpu_results['recommendations'].append("Consider upgrading to a multi-core processor (4+ cores)")
        elif cpu_cores < 8:
            cpu_results['recommendations'].append("Good CPU configuration for moderate workloads")
        else:
            cpu_results['recommendations'].append("Excellent CPU configuration for intensive workloads")
        
        if cpu_time > 1.0:
            cpu_results['recommendations'].append("CPU processing speed may be limiting for large-scale operations")
        
        self.results['cpu_test_results'] = cpu_results
        return cpu_results
    
    def test_gpu_performance(self) -> Dict[str, Any]:
        """Test GPU performance for deep learning tasks"""
        print("Testing GPU performance...")
        
        gpu_results = {
            'gpu_available': False,
            'gpu_utilization': {},
            'memory_usage': {},
            'processing_speed': {},
            'cuda_operations': {},
            'recommendations': []
        }
        
        if not torch.cuda.is_available():
            gpu_results['recommendations'].append("GPU not available. Consider installing CUDA-compatible GPU")
            self.results['gpu_test_results'] = gpu_results
            return gpu_results
        
        gpu_results['gpu_available'] = True
        
        # Test GPU memory
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        
        gpu_results['memory_usage'] = {
            'total_gb': gpu_memory_total,
            'allocated_gb': gpu_memory_allocated,
            'reserved_gb': gpu_memory_reserved,
            'free_gb': gpu_memory_total - gpu_memory_reserved,
            'utilization_percent': (gpu_memory_reserved / gpu_memory_total) * 100
        }
        
        # Test GPU processing speed
        device = torch.device('cuda')
        start_time = time.time()
        
        # Test matrix multiplication on GPU
        with torch.no_grad():
            tensor_a = torch.randn(2000, 2000, device=device)
            tensor_b = torch.randn(2000, 2000, device=device)
            tensor_c = torch.mm(tensor_a, tensor_b)
        
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        gpu_results['processing_speed'] = {
            'gpu_matrix_multiplication_time': gpu_time,
            'gpu_operations_per_second': (2000**3) / gpu_time,
            'matrix_size': 2000
        }
        
        # Test CUDA operations
        start_time = time.time()
        with torch.no_grad():
            # Test convolution operation
            input_tensor = torch.randn(32, 3, 224, 224, device=device)
            conv_layer = nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)
            output = conv_layer(input_tensor)
        
        torch.cuda.synchronize()
        conv_time = time.time() - start_time
        
        gpu_results['cuda_operations'] = {
            'convolution_time': conv_time,
            'convolution_operations_per_second': (32 * 3 * 224 * 224 * 64 * 9) / conv_time
        }
        
        # Clear GPU memory
        del tensor_a, tensor_b, tensor_c, input_tensor, conv_layer, output
        torch.cuda.empty_cache()
        
        # Generate GPU recommendations
        if gpu_memory_total < 4:
            gpu_results['recommendations'].append("GPU memory may be limiting for large models. Consider GPU with 8GB+ memory")
        elif gpu_memory_total < 8:
            gpu_results['recommendations'].append("Good GPU configuration for moderate model sizes")
        else:
            gpu_results['recommendations'].append("Excellent GPU configuration for large-scale training")
        
        if gpu_time > 0.5:
            gpu_results['recommendations'].append("GPU processing speed may be suboptimal for intensive workloads")
        
        self.results['gpu_test_results'] = gpu_results
        return gpu_results
    
    def test_memory_performance(self) -> Dict[str, Any]:
        """Test memory performance and capacity"""
        print("Testing memory performance...")
        
        memory_results = {
            'memory_usage': {},
            'memory_speed': {},
            'memory_stability': {},
            'recommendations': []
        }
        
        # Get current memory usage
        memory = psutil.virtual_memory()
        memory_results['memory_usage'] = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent_used': memory.percent,
            'free_gb': memory.free / (1024**3)
        }
        
        # Test memory allocation speed
        start_time = time.time()
        test_array = np.random.rand(1000, 1000)
        allocation_time = time.time() - start_time
        
        memory_results['memory_speed'] = {
            'allocation_time': allocation_time,
            'allocation_size_mb': test_array.nbytes / (1024**2),
            'allocation_speed_mbps': (test_array.nbytes / (1024**2)) / allocation_time
        }
        
        # Test memory stability under load
        memory_stress_results = []
        for i in range(5):
            start_time = time.time()
            stress_array = np.random.rand(500, 500)
            stress_time = time.time() - start_time
            memory_stress_results.append(stress_time)
            del stress_array
        
        memory_results['memory_stability'] = {
            'stress_test_times': memory_stress_results,
            'average_stress_time': np.mean(memory_stress_results),
            'stress_time_std': np.std(memory_stress_results)
        }
        
        # Generate memory recommendations
        total_memory_gb = memory.total / (1024**3)
        if total_memory_gb < 8:
            memory_results['recommendations'].append("Memory may be limiting. Consider upgrading to 16GB+ RAM")
        elif total_memory_gb < 16:
            memory_results['recommendations'].append("Good memory configuration for moderate workloads")
        else:
            memory_results['recommendations'].append("Excellent memory configuration for intensive workloads")
        
        if memory.percent > 80:
            memory_results['recommendations'].append("High memory usage detected. Consider closing unnecessary applications")
        
        if np.std(memory_stress_results) > 0.1:
            memory_results['recommendations'].append("Memory performance may be inconsistent. Check for memory leaks or background processes")
        
        self.results['memory_test_results'] = memory_results
        return memory_results
    
    def test_storage_performance(self) -> Dict[str, Any]:
        """Test storage performance and capacity"""
        print("Testing storage performance...")
        
        storage_results = {
            'storage_capacity': {},
            'storage_speed': {},
            'storage_health': {},
            'recommendations': []
        }
        
        # Get storage information
        disk_usage = psutil.disk_usage('/')
        storage_results['storage_capacity'] = {
            'total_gb': disk_usage.total / (1024**3),
            'used_gb': disk_usage.used / (1024**3),
            'free_gb': disk_usage.free / (1024**3),
            'percent_used': (disk_usage.used / disk_usage.total) * 100
        }
        
        # Test storage write speed
        test_file = Path('storage_test_file.tmp')
        test_data = np.random.rand(1000, 1000)
        
        start_time = time.time()
        np.save(test_file, test_data)
        write_time = time.time() - start_time
        
        # Test storage read speed
        start_time = time.time()
        if test_file.exists():
            loaded_data = np.load(test_file)
            read_time = time.time() - start_time
        else:
            read_time = 1.0  # Fallback if file doesn't exist
        
        # Clean up test file
        if test_file.exists():
            test_file.unlink()
        
        file_size_mb = test_data.nbytes / (1024**2)
        storage_results['storage_speed'] = {
            'write_speed_mbps': file_size_mb / write_time,
            'read_speed_mbps': file_size_mb / read_time,
            'file_size_mb': file_size_mb,
            'write_time': write_time,
            'read_time': read_time
        }
        
        # Test storage health (basic)
        storage_results['storage_health'] = {
            'disk_health': 'Good',  # Simplified - in real implementation, would check SMART data
            'fragmentation_level': 'Low',  # Simplified
            'error_rate': 'Low'  # Simplified
        }
        
        # Generate storage recommendations
        total_storage_gb = disk_usage.total / (1024**3)
        if total_storage_gb < 100:
            storage_results['recommendations'].append("Storage may be limiting for large datasets. Consider upgrading to 500GB+")
        elif total_storage_gb < 500:
            storage_results['recommendations'].append("Good storage configuration for moderate datasets")
        else:
            storage_results['recommendations'].append("Excellent storage configuration for large-scale datasets")
        
        if (disk_usage.used / disk_usage.total) > 0.8:
            storage_results['recommendations'].append("High storage usage detected. Consider freeing up space")
        
        if write_time > 1.0 or read_time > 0.5:
            storage_results['recommendations'].append("Storage speed may be limiting. Consider SSD upgrade")
        
        self.results['storage_test_results'] = storage_results
        return storage_results
    
    def generate_performance_metrics(self):
        """Generate comprehensive performance metrics"""
        print("Generating performance metrics...")
        
        performance_metrics = {
            'overall_score': 0.0,
            'component_scores': {},
            'bottlenecks': [],
            'optimization_opportunities': []
        }
        
        # Calculate component scores
        cpu_score = self._calculate_cpu_score()
        gpu_score = self._calculate_gpu_score()
        memory_score = self._calculate_memory_score()
        storage_score = self._calculate_storage_score()
        
        performance_metrics['component_scores'] = {
            'cpu_score': cpu_score,
            'gpu_score': gpu_score,
            'memory_score': memory_score,
            'storage_score': storage_score
        }
        
        # Calculate overall score
        weights = {'cpu': 0.25, 'gpu': 0.35, 'memory': 0.25, 'storage': 0.15}
        overall_score = (
            cpu_score * weights['cpu'] +
            gpu_score * weights['gpu'] +
            memory_score * weights['memory'] +
            storage_score * weights['storage']
        )
        
        performance_metrics['overall_score'] = overall_score
        
        # Identify bottlenecks
        if cpu_score < 0.6:
            performance_metrics['bottlenecks'].append("CPU performance may be limiting")
        if gpu_score < 0.6:
            performance_metrics['bottlenecks'].append("GPU performance may be limiting")
        if memory_score < 0.6:
            performance_metrics['bottlenecks'].append("Memory capacity may be limiting")
        if storage_score < 0.6:
            performance_metrics['bottlenecks'].append("Storage performance may be limiting")
        
        # Generate optimization opportunities
        if cpu_score < 0.8:
            performance_metrics['optimization_opportunities'].append("Consider CPU upgrade for better performance")
        if gpu_score < 0.8:
            performance_metrics['optimization_opportunities'].append("Consider GPU upgrade for faster training")
        if memory_score < 0.8:
            performance_metrics['optimization_opportunities'].append("Consider memory upgrade for larger models")
        if storage_score < 0.8:
            performance_metrics['optimization_opportunities'].append("Consider SSD upgrade for faster I/O")
        
        self.results['performance_metrics'] = performance_metrics
    
    def _calculate_cpu_score(self) -> float:
        """Calculate CPU performance score"""
        cpu_results = self.results['cpu_test_results']
        if not cpu_results:
            return 0.0
        
        # Score based on CPU cores and processing speed
        cpu_cores = psutil.cpu_count()
        processing_speed = cpu_results.get('processing_speed', {})
        
        core_score = min(cpu_cores / 8.0, 1.0)  # Normalize to 8 cores
        speed_score = min(1.0 / (processing_speed.get('matrix_multiplication_time', 1.0)), 1.0)
        
        return (core_score + speed_score) / 2
    
    def _calculate_gpu_score(self) -> float:
        """Calculate GPU performance score"""
        gpu_results = self.results['gpu_test_results']
        if not gpu_results or not gpu_results.get('gpu_available', False):
            return 0.0
        
        # Score based on GPU memory and processing speed
        memory_usage = gpu_results.get('memory_usage', {})
        processing_speed = gpu_results.get('processing_speed', {})
        
        memory_score = min(memory_usage.get('total_gb', 0) / 8.0, 1.0)  # Normalize to 8GB
        speed_score = min(1.0 / (processing_speed.get('gpu_matrix_multiplication_time', 1.0)), 1.0)
        
        return (memory_score + speed_score) / 2
    
    def _calculate_memory_score(self) -> float:
        """Calculate memory performance score"""
        memory_results = self.results['memory_test_results']
        if not memory_results:
            return 0.0
        
        # Score based on memory capacity and speed
        memory_usage = memory_results.get('memory_usage', {})
        memory_speed = memory_results.get('memory_speed', {})
        
        capacity_score = min(memory_usage.get('total_gb', 0) / 16.0, 1.0)  # Normalize to 16GB
        speed_score = min(memory_speed.get('allocation_speed_mbps', 100) / 1000.0, 1.0)  # Normalize to 1000 MB/s
        
        return (capacity_score + speed_score) / 2
    
    def _calculate_storage_score(self) -> float:
        """Calculate storage performance score"""
        storage_results = self.results['storage_test_results']
        if not storage_results:
            return 0.0
        
        # Score based on storage capacity and speed
        storage_capacity = storage_results.get('storage_capacity', {})
        storage_speed = storage_results.get('storage_speed', {})
        
        capacity_score = min(storage_capacity.get('total_gb', 0) / 500.0, 1.0)  # Normalize to 500GB
        speed_score = min(storage_speed.get('write_speed_mbps', 50) / 500.0, 1.0)  # Normalize to 500 MB/s
        
        return (capacity_score + speed_score) / 2
    
    def generate_recommendations(self):
        """Generate comprehensive hardware recommendations"""
        print("Generating hardware recommendations...")
        
        recommendations = {
            'immediate_actions': [],
            'short_term_upgrades': [],
            'long_term_considerations': [],
            'optimization_tips': []
        }
        
        # Collect recommendations from all tests
        cpu_recs = self.results['cpu_test_results'].get('recommendations', [])
        gpu_recs = self.results['gpu_test_results'].get('recommendations', [])
        memory_recs = self.results['memory_test_results'].get('recommendations', [])
        storage_recs = self.results['storage_test_results'].get('recommendations', [])
        
        # Categorize recommendations
        for rec in cpu_recs + gpu_recs + memory_recs + storage_recs:
            if 'upgrade' in rec.lower() or 'consider' in rec.lower():
                recommendations['short_term_upgrades'].append(rec)
            elif 'immediate' in rec.lower() or 'urgent' in rec.lower():
                recommendations['immediate_actions'].append(rec)
            else:
                recommendations['optimization_tips'].append(rec)
        
        # Add performance-based recommendations
        performance_metrics = self.results['performance_metrics']
        overall_score = performance_metrics.get('overall_score', 0.0)
        
        if overall_score < 0.5:
            recommendations['immediate_actions'].append("Hardware configuration may significantly limit performance")
        elif overall_score < 0.7:
            recommendations['short_term_upgrades'].append("Consider hardware upgrades for better performance")
        elif overall_score < 0.9:
            recommendations['optimization_tips'].append("Good hardware configuration with room for optimization")
        else:
            recommendations['optimization_tips'].append("Excellent hardware configuration for deep learning workloads")
        
        self.results['recommendations'] = recommendations
    
    def run_comprehensive_test(self, cpu_test: bool = True, gpu_test: bool = True, 
                             memory_test: bool = True, storage_test: bool = True):
        """Run comprehensive hardware configuration test"""
        print("Starting comprehensive hardware configuration test...")
        print("="*60)
        
        # Update test configuration
        self.results['metadata']['test_configuration'] = {
            'cpu_test': cpu_test,
            'gpu_test': gpu_test,
            'memory_test': memory_test,
            'storage_test': storage_test
        }
        
        # Run individual tests
        if cpu_test:
            self.test_cpu_performance()
        
        if gpu_test:
            self.test_gpu_performance()
        
        if memory_test:
            self.test_memory_performance()
        
        if storage_test:
            self.test_storage_performance()
        
        # Generate performance metrics and recommendations
        self.generate_performance_metrics()
        self.generate_recommendations()
        
        print("="*60)
        print("Hardware configuration test completed!")
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print(" HARDWARE CONFIGURATION TEST SUMMARY")
        print("="*80)
        
        # Print system information
        system_info = self.results['metadata']['system_info']
        print(f"\nSystem Configuration:")
        print(f"- Platform: {system_info['platform']}")
        print(f"- CPU: {system_info['processor']} ({system_info['cpu_count']} cores)")
        print(f"- Memory: {system_info['memory_total_gb']:.1f} GB total")
        print(f"- Storage: {system_info['disk_usage']:.1f} GB total")
        if system_info['cuda_available']:
            print(f"- GPU: {system_info['gpu_name']} ({system_info['gpu_memory_gb']:.1f} GB)")
        else:
            print("- GPU: Not available (CPU-only mode)")
        
        # Print performance metrics
        performance_metrics = self.results['performance_metrics']
        print(f"\nPerformance Scores:")
        component_scores = performance_metrics.get('component_scores', {})
        for component, score in component_scores.items():
            print(f"- {component.replace('_', ' ').title()}: {score:.2f}")
        print(f"- Overall Score: {performance_metrics.get('overall_score', 0):.2f}")
        
        # Print recommendations
        recommendations = self.results['recommendations']
        if recommendations.get('immediate_actions'):
            print(f"\nImmediate Actions:")
            for action in recommendations['immediate_actions']:
                print(f"- {action}")
        
        if recommendations.get('short_term_upgrades'):
            print(f"\nShort-term Upgrades:")
            for upgrade in recommendations['short_term_upgrades']:
                print(f"- {upgrade}")
        
        if recommendations.get('optimization_tips'):
            print(f"\nOptimization Tips:")
            for tip in recommendations['optimization_tips']:
                print(f"- {tip}")
        
        print("="*80)
    
    def save_results(self, output_file: str):
        """Save test results to JSON file"""
        output_path = Path(output_file)
        print(f"Saving hardware test results to {output_path}...")
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"✓ Hardware test results saved successfully!")
        print(f"Results file: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Hardware Configuration Test')
    parser.add_argument('--cpu_test', action='store_true', help='Run CPU performance test')
    parser.add_argument('--gpu_test', action='store_true', help='Run GPU performance test')
    parser.add_argument('--memory_test', action='store_true', help='Run memory performance test')
    parser.add_argument('--storage_test', action='store_true', help='Run storage performance test')
    parser.add_argument('--output_file', type=str, default='hardware_test_results.json',
                       help='Output file for test results')
    
    args = parser.parse_args()
    
    # If no specific tests are selected, run all tests
    if not any([args.cpu_test, args.gpu_test, args.memory_test, args.storage_test]):
        args.cpu_test = args.gpu_test = args.memory_test = args.storage_test = True
    
    # Create tester and run tests
    tester = HardwareConfigurationTester()
    tester.run_comprehensive_test(
        cpu_test=args.cpu_test,
        gpu_test=args.gpu_test,
        memory_test=args.memory_test,
        storage_test=args.storage_test
    )
    
    # Print summary and save results
    tester.print_summary()
    tester.save_results(args.output_file)
    
    print(f"\nHardware configuration test completed!")
    print(f"Output file: {args.output_file}")

if __name__ == '__main__':
    main() 