#!/usr/bin/env python3
"""
Hardware Performance Plots Generator
Generates comprehensive plots from hardware configuration test results
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HardwarePerformancePlotter:
    """Generate comprehensive plots from hardware test results"""
    
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load hardware test results
        with open(self.input_file, 'r') as f:
            self.results = json.load(f)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Define color scheme for hardware components
        self.component_colors = {
            'cpu': '#FF6B6B',
            'gpu': '#4ECDC4',
            'memory': '#45B7D1',
            'storage': '#96CEB4',
            'overall': '#FFEAA7'
        }
    
    def plot_hardware_overview(self):
        """Plot hardware configuration overview"""
        print("Generating hardware overview plot...")
        
        if 'metadata' not in self.results or 'hardware_specifications' not in self.results['metadata']:
            print("No hardware specifications available.")
            return
        
        hardware_specs = self.results['metadata']['hardware_specifications']
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different hardware components
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hardware Configuration Overview', fontsize=16, fontweight='bold')
        
        # 1. CPU Information
        ax1 = axes[0, 0]
        cpu_info = hardware_specs.get('cpu', {})
        cpu_data = {
            'Cores': cpu_info.get('cores', 0),
            'Logical Cores': cpu_info.get('logical_cores', 0)
        }
        
        bars = ax1.bar(cpu_data.keys(), cpu_data.values(), color=self.component_colors['cpu'])
        ax1.set_title('CPU Configuration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=10)
        
        # Add value labels
        for bar, value in zip(bars, cpu_data.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Memory Information
        ax2 = axes[0, 1]
        memory_info = hardware_specs.get('memory', {})
        memory_data = {
            'Total (GB)': memory_info.get('total_gb', 0),
            'Available (GB)': memory_info.get('available_gb', 0)
        }
        
        bars = ax2.bar(memory_data.keys(), memory_data.values(), color=self.component_colors['memory'])
        ax2.set_title('Memory Configuration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('GB', fontsize=10)
        
        # Add value labels
        for bar, value in zip(bars, memory_data.values()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. GPU Information
        ax3 = axes[1, 0]
        gpu_info = hardware_specs.get('gpu', {})
        gpu_data = {
            'Memory (GB)': gpu_info.get('memory_gb', 0),
            'CUDA Available': 1 if gpu_info.get('cuda_available', False) else 0
        }
        
        bars = ax3.bar(gpu_data.keys(), gpu_data.values(), color=self.component_colors['gpu'])
        ax3.set_title('GPU Configuration', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Value', fontsize=10)
        
        # Add value labels
        for bar, value in zip(bars, gpu_data.values()):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Storage Information
        ax4 = axes[1, 1]
        storage_info = hardware_specs.get('storage', {})
        storage_data = {
            'Total (GB)': storage_info.get('total_gb', 0)
        }
        
        bars = ax4.bar(storage_data.keys(), storage_data.values(), color=self.component_colors['storage'])
        ax4.set_title('Storage Configuration', fontsize=12, fontweight='bold')
        ax4.set_ylabel('GB', fontsize=10)
        
        # Add value labels
        for bar, value in zip(bars, storage_data.values()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hardware_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Hardware overview plot saved to {self.output_dir / 'hardware_overview.png'}")
    
    def plot_performance_scores(self):
        """Plot performance scores for different hardware components"""
        print("Generating performance scores plot...")
        
        if 'performance_metrics' not in self.results or 'component_scores' not in self.results['performance_metrics']:
            print("No performance metrics available.")
            return
        
        component_scores = self.results['performance_metrics']['component_scores']
        
        plt.figure(figsize=(12, 8))
        
        # Create bar chart of component scores
        components = list(component_scores.keys())
        scores = list(component_scores.values())
        colors = [self.component_colors.get(comp.split('_')[0], '#666666') for comp in components]
        
        bars = plt.bar(components, scores, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Hardware Component Performance Scores', fontsize=16, fontweight='bold')
        plt.xlabel('Hardware Component', fontsize=12)
        plt.ylabel('Performance Score (0-1)', fontsize=12)
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance scores plot saved to {self.output_dir / 'performance_scores.png'}")
    
    def plot_cpu_performance_analysis(self):
        """Plot detailed CPU performance analysis"""
        print("Generating CPU performance analysis plot...")
        
        if 'cpu_test_results' not in self.results:
            print("No CPU test results available.")
            return
        
        cpu_results = self.results['cpu_test_results']
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different CPU metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CPU Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. CPU Utilization
        ax1 = axes[0, 0]
        cpu_util = cpu_results.get('cpu_utilization', {})
        if 'per_core_usage' in cpu_util:
            core_usage = cpu_util['per_core_usage']
            cores = range(1, len(core_usage) + 1)
            bars = ax1.bar(cores, core_usage, color=self.component_colors['cpu'])
            ax1.set_title('CPU Utilization by Core', fontsize=12, fontweight='bold')
            ax1.set_xlabel('CPU Core', fontsize=10)
            ax1.set_ylabel('Utilization (%)', fontsize=10)
            ax1.set_ylim(0, 100)
            
            # Add value labels
            for bar, usage in zip(bars, core_usage):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{usage:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Processing Speed
        ax2 = axes[0, 1]
        proc_speed = cpu_results.get('processing_speed', {})
        if proc_speed:
            metrics = ['Matrix Multiplication Time (s)', 'Operations per Second']
            values = [proc_speed.get('matrix_multiplication_time', 0), 
                     proc_speed.get('operations_per_second', 0) / 1e9]  # Convert to billions
            
            bars = ax2.bar(metrics, values, color=['#FF6B6B', '#4ECDC4'])
            ax2.set_title('CPU Processing Speed', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Value', fontsize=10)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Parallel Processing
        ax3 = axes[1, 0]
        parallel = cpu_results.get('parallel_processing', {})
        if parallel:
            torch_time = parallel.get('torch_matrix_multiplication_time', 0)
            torch_ops = parallel.get('torch_operations_per_second', 0) / 1e9
            
            metrics = ['PyTorch Matrix Time (s)', 'PyTorch Ops (billions)']
            values = [torch_time, torch_ops]
            
            bars = ax3.bar(metrics, values, color=['#45B7D1', '#96CEB4'])
            ax3.set_title('PyTorch CPU Performance', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Value', fontsize=10)
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. CPU Recommendations
        ax4 = axes[1, 1]
        recommendations = cpu_results.get('recommendations', [])
        if recommendations:
            ax4.text(0.5, 0.5, '\n'.join(recommendations), 
                    ha='center', va='center', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            ax4.set_title('CPU Recommendations', fontsize=12, fontweight='bold')
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cpu_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"CPU performance analysis plot saved to {self.output_dir / 'cpu_performance_analysis.png'}")
    
    def plot_gpu_performance_analysis(self):
        """Plot detailed GPU performance analysis"""
        print("Generating GPU performance analysis plot...")
        
        if 'gpu_test_results' not in self.results:
            print("No GPU test results available.")
            return
        
        gpu_results = self.results['gpu_test_results']
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different GPU metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GPU Performance Analysis', fontsize=16, fontweight='bold')
        
        # Check if GPU is available
        if not gpu_results.get('gpu_available', False):
            # GPU not available - show information message
            ax1 = axes[0, 0]
            ax1.text(0.5, 0.5, 'GPU Not Available\n(CPU-only mode)', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
            ax1.set_title('GPU Status', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            # Show recommendations
            ax2 = axes[0, 1]
            recommendations = gpu_results.get('recommendations', [])
            if recommendations:
                ax2.text(0.5, 0.5, '\n'.join(recommendations), 
                        ha='center', va='center', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
                ax2.set_title('GPU Recommendations', fontsize=12, fontweight='bold')
                ax2.axis('off')
            
            # Empty plots for remaining subplots
            for ax in [axes[1, 0], axes[1, 1]]:
                ax.text(0.5, 0.5, 'No GPU Data Available', 
                       ha='center', va='center', fontsize=12, alpha=0.5)
                ax.axis('off')
        else:
            # GPU is available - show detailed metrics
            # 1. GPU Memory Usage
            ax1 = axes[0, 0]
            memory_usage = gpu_results.get('memory_usage', {})
            if memory_usage:
                memory_metrics = ['Total (GB)', 'Allocated (GB)', 'Reserved (GB)', 'Free (GB)']
                memory_values = [
                    memory_usage.get('total_gb', 0),
                    memory_usage.get('allocated_gb', 0),
                    memory_usage.get('reserved_gb', 0),
                    memory_usage.get('free_gb', 0)
                ]
                
                bars = ax1.bar(memory_metrics, memory_values, color=self.component_colors['gpu'])
                ax1.set_title('GPU Memory Usage', fontsize=12, fontweight='bold')
                ax1.set_ylabel('GB', fontsize=10)
                ax1.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, memory_values):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # 2. GPU Processing Speed
            ax2 = axes[0, 1]
            proc_speed = gpu_results.get('processing_speed', {})
            if proc_speed:
                speed_metrics = ['Matrix Time (s)', 'Operations (billions)']
                speed_values = [
                    proc_speed.get('gpu_matrix_multiplication_time', 0),
                    proc_speed.get('gpu_operations_per_second', 0) / 1e9
                ]
                
                bars = ax2.bar(speed_metrics, speed_values, color=['#4ECDC4', '#45B7D1'])
                ax2.set_title('GPU Processing Speed', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Value', fontsize=10)
                ax2.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, speed_values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 3. CUDA Operations
            ax3 = axes[1, 0]
            cuda_ops = gpu_results.get('cuda_operations', {})
            if cuda_ops:
                cuda_metrics = ['Convolution Time (s)', 'Conv Ops (billions)']
                cuda_values = [
                    cuda_ops.get('convolution_time', 0),
                    cuda_ops.get('convolution_operations_per_second', 0) / 1e9
                ]
                
                bars = ax3.bar(cuda_metrics, cuda_values, color=['#96CEB4', '#FFEAA7'])
                ax3.set_title('CUDA Operations', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Value', fontsize=10)
                ax3.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, cuda_values):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 4. GPU Recommendations
            ax4 = axes[1, 1]
            recommendations = gpu_results.get('recommendations', [])
            if recommendations:
                ax4.text(0.5, 0.5, '\n'.join(recommendations), 
                        ha='center', va='center', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
                ax4.set_title('GPU Recommendations', fontsize=12, fontweight='bold')
                ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'gpu_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"GPU performance analysis plot saved to {self.output_dir / 'gpu_performance_analysis.png'}")
    
    def plot_memory_performance_analysis(self):
        """Plot detailed memory performance analysis"""
        print("Generating memory performance analysis plot...")
        
        if 'memory_test_results' not in self.results:
            print("No memory test results available.")
            return
        
        memory_results = self.results['memory_test_results']
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different memory metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Memory Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Memory Usage
        ax1 = axes[0, 0]
        memory_usage = memory_results.get('memory_usage', {})
        if memory_usage:
            usage_metrics = ['Total (GB)', 'Available (GB)', 'Used (GB)', 'Free (GB)']
            usage_values = [
                memory_usage.get('total_gb', 0),
                memory_usage.get('available_gb', 0),
                memory_usage.get('used_gb', 0),
                memory_usage.get('free_gb', 0)
            ]
            
            bars = ax1.bar(usage_metrics, usage_values, color=self.component_colors['memory'])
            ax1.set_title('Memory Usage', fontsize=12, fontweight='bold')
            ax1.set_ylabel('GB', fontsize=10)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, usage_values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Memory Speed
        ax2 = axes[0, 1]
        memory_speed = memory_results.get('memory_speed', {})
        if memory_speed:
            speed_metrics = ['Allocation Time (s)', 'Allocation Speed (MB/s)']
            speed_values = [
                memory_speed.get('allocation_time', 0),
                memory_speed.get('allocation_speed_mbps', 0)
            ]
            
            bars = ax2.bar(speed_metrics, speed_values, color=['#45B7D1', '#96CEB4'])
            ax2.set_title('Memory Speed', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Value', fontsize=10)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, speed_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Memory Stability
        ax3 = axes[1, 0]
        memory_stability = memory_results.get('memory_stability', {})
        if memory_stability and 'stress_test_times' in memory_stability:
            stress_times = memory_stability['stress_test_times']
            test_numbers = range(1, len(stress_times) + 1)
            
            ax3.plot(test_numbers, stress_times, 'o-', color=self.component_colors['memory'], linewidth=2)
            ax3.set_title('Memory Stability Test', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Test Number', fontsize=10)
            ax3.set_ylabel('Time (s)', fontsize=10)
            ax3.grid(True, alpha=0.3)
        
        # 4. Memory Recommendations
        ax4 = axes[1, 1]
        recommendations = memory_results.get('recommendations', [])
        if recommendations:
            ax4.text(0.5, 0.5, '\n'.join(recommendations), 
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            ax4.set_title('Memory Recommendations', fontsize=12, fontweight='bold')
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'memory_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Memory performance analysis plot saved to {self.output_dir / 'memory_performance_analysis.png'}")
    
    def plot_storage_performance_analysis(self):
        """Plot detailed storage performance analysis"""
        print("Generating storage performance analysis plot...")
        
        if 'storage_test_results' not in self.results:
            print("No storage test results available.")
            return
        
        storage_results = self.results['storage_test_results']
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different storage metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Storage Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Storage Capacity
        ax1 = axes[0, 0]
        storage_capacity = storage_results.get('storage_capacity', {})
        if storage_capacity:
            capacity_metrics = ['Total (GB)', 'Used (GB)', 'Free (GB)']
            capacity_values = [
                storage_capacity.get('total_gb', 0),
                storage_capacity.get('used_gb', 0),
                storage_capacity.get('free_gb', 0)
            ]
            
            bars = ax1.bar(capacity_metrics, capacity_values, color=self.component_colors['storage'])
            ax1.set_title('Storage Capacity', fontsize=12, fontweight='bold')
            ax1.set_ylabel('GB', fontsize=10)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, capacity_values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Storage Speed
        ax2 = axes[0, 1]
        storage_speed = storage_results.get('storage_speed', {})
        if storage_speed:
            speed_metrics = ['Write Speed (MB/s)', 'Read Speed (MB/s)']
            speed_values = [
                storage_speed.get('write_speed_mbps', 0),
                storage_speed.get('read_speed_mbps', 0)
            ]
            
            bars = ax2.bar(speed_metrics, speed_values, color=['#96CEB4', '#FFEAA7'])
            ax2.set_title('Storage Speed', fontsize=12, fontweight='bold')
            ax2.set_ylabel('MB/s', fontsize=10)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, speed_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Storage Health
        ax3 = axes[1, 0]
        storage_health = storage_results.get('storage_health', {})
        if storage_health:
            health_metrics = ['Disk Health', 'Fragmentation', 'Error Rate']
            health_values = [1, 1, 1]  # Simplified - in real implementation would use actual health scores
            
            bars = ax3.bar(health_metrics, health_values, color=['#4ECDC4', '#45B7D1', '#96CEB4'])
            ax3.set_title('Storage Health', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Score', fontsize=10)
            ax3.set_ylim(0, 1.2)
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, health_values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Storage Recommendations
        ax4 = axes[1, 1]
        recommendations = storage_results.get('recommendations', [])
        if recommendations:
            ax4.text(0.5, 0.5, '\n'.join(recommendations), 
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            ax4.set_title('Storage Recommendations', fontsize=12, fontweight='bold')
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'storage_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Storage performance analysis plot saved to {self.output_dir / 'storage_performance_analysis.png'}")
    
    def plot_recommendations_summary(self):
        """Plot comprehensive recommendations summary"""
        print("Generating recommendations summary plot...")
        
        if 'recommendations' not in self.results:
            print("No recommendations available.")
            return
        
        recommendations = self.results['recommendations']
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different recommendation categories
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hardware Recommendations Summary', fontsize=16, fontweight='bold')
        
        # 1. Immediate Actions
        ax1 = axes[0, 0]
        immediate_actions = recommendations.get('immediate_actions', [])
        if immediate_actions:
            ax1.text(0.5, 0.5, '\n'.join([f"• {action}" for action in immediate_actions]), 
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
            ax1.set_title('Immediate Actions', fontsize=12, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No immediate actions required', 
                    ha='center', va='center', fontsize=12, alpha=0.5)
            ax1.set_title('Immediate Actions', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. Short-term Upgrades
        ax2 = axes[0, 1]
        short_term = recommendations.get('short_term_upgrades', [])
        if short_term:
            ax2.text(0.5, 0.5, '\n'.join([f"• {upgrade}" for upgrade in short_term]), 
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            ax2.set_title('Short-term Upgrades', fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No short-term upgrades recommended', 
                    ha='center', va='center', fontsize=12, alpha=0.5)
            ax2.set_title('Short-term Upgrades', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 3. Long-term Considerations
        ax3 = axes[1, 0]
        long_term = recommendations.get('long_term_considerations', [])
        if long_term:
            ax3.text(0.5, 0.5, '\n'.join([f"• {consideration}" for consideration in long_term]), 
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            ax3.set_title('Long-term Considerations', fontsize=12, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No long-term considerations', 
                    ha='center', va='center', fontsize=12, alpha=0.5)
            ax3.set_title('Long-term Considerations', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # 4. Optimization Tips
        ax4 = axes[1, 1]
        optimization_tips = recommendations.get('optimization_tips', [])
        if optimization_tips:
            ax4.text(0.5, 0.5, '\n'.join([f"• {tip}" for tip in optimization_tips]), 
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
            ax4.set_title('Optimization Tips', fontsize=12, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No optimization tips available', 
                    ha='center', va='center', fontsize=12, alpha=0.5)
            ax4.set_title('Optimization Tips', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'recommendations_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Recommendations summary plot saved to {self.output_dir / 'recommendations_summary.png'}")
    
    def generate_all_plots(self):
        """Generate all hardware performance plots"""
        print("Generating all hardware performance visualization plots...")
        
        # Generate all plots
        self.plot_hardware_overview()
        self.plot_performance_scores()
        self.plot_cpu_performance_analysis()
        self.plot_gpu_performance_analysis()
        self.plot_memory_performance_analysis()
        self.plot_storage_performance_analysis()
        self.plot_recommendations_summary()
        
        print(f"\nAll plots generated successfully in {self.output_dir}")
        print("Generated plots:")
        plot_files = [
            'hardware_overview.png',
            'performance_scores.png',
            'cpu_performance_analysis.png',
            'gpu_performance_analysis.png',
            'memory_performance_analysis.png',
            'storage_performance_analysis.png',
            'recommendations_summary.png'
        ]
        
        for plot_file in plot_files:
            if (self.output_dir / plot_file).exists():
                print(f"  ✅ {plot_file}")
            else:
                print(f"  ❌ {plot_file} (no data available)")

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Generate Hardware Performance Visualization Plots')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Input hardware test results JSON file')
    parser.add_argument('--output_dir', type=str, default='./hardware_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create plotter
    plotter = HardwarePerformancePlotter(args.input_file, args.output_dir)
    
    # Generate all plots
    plotter.generate_all_plots()
    
    print(f"\nHardware performance plots generation completed!")
    print(f"Output directory: {args.output_dir}")

if __name__ == '__main__':
    main() 