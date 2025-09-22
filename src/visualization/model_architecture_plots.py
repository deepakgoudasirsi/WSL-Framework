#!/usr/bin/env python3
"""
Model Architecture Plots Generator
Generates comprehensive plots from model architecture analysis results
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

class ModelArchitecturePlotter:
    """Generate comprehensive plots from model architecture analysis results"""
    
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model architecture results
        with open(self.input_file, 'r') as f:
            self.results = json.load(f)
        
        # Convert to DataFrame for easier plotting
        if 'model_architecture_results' in self.results:
            self.df = pd.DataFrame(self.results['model_architecture_results'])
        else:
            self.df = pd.DataFrame()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Define color scheme for model types
        self.model_colors = {
            'simple_cnn': '#FF6B6B',
            'robust_cnn': '#4ECDC4',
            'resnet': '#45B7D1',
            'robust_resnet': '#96CEB4',
            'mlp': '#FFEAA7',
            'robust_mlp': '#DDA0DD'
        }
        
        # Define color scheme for datasets
        self.dataset_colors = {
            'cifar10': '#FF6B6B',
            'mnist': '#4ECDC4'
        }
    
    def plot_architecture_overview(self):
        """Plot model architecture overview"""
        print("Generating architecture overview plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run model architecture analysis first.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different architecture metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Architecture Overview', fontsize=16, fontweight='bold')
        
        # 1. Parameter Count Comparison
        ax1 = axes[0, 0]
        models = self.df['model_type']
        parameters = self.df['total_parameters']
        
        bars = ax1.bar(models, parameters, 
                       color=[self.model_colors.get(m.lower(), '#666666') for m in models])
        ax1.set_title('Total Parameters by Model Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Parameters', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, param_count in zip(bars, parameters):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100000,
                    f'{param_count:,}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 2. Memory Usage Comparison
        ax2 = axes[0, 1]
        memory_usage = self.df['memory_usage_gb']
        
        bars = ax2.bar(models, memory_usage, 
                       color=[self.model_colors.get(m.lower(), '#666666') for m in models])
        ax2.set_title('Memory Usage by Model Type', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Memory Usage (GB)', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, memory in zip(bars, memory_usage):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{memory:.1f} GB', ha='center', va='bottom', fontweight='bold')
        
        # 3. Training Time Factor Comparison
        ax3 = axes[1, 0]
        training_times = self.df['training_time_factor']
        
        bars = ax3.bar(models, training_times, 
                       color=[self.model_colors.get(m.lower(), '#666666') for m in models])
        ax3.set_title('Training Time Factor by Model Type', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Training Time Factor (x)', fontsize=10)
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, time_factor in zip(bars, training_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{time_factor:.0f}x', ha='center', va='bottom', fontweight='bold')
        
        # 4. Complexity Factor Comparison
        ax4 = axes[1, 1]
        complexity_factors = self.df['complexity_factor']
        
        bars = ax4.bar(models, complexity_factors, 
                       color=[self.model_colors.get(m.lower(), '#666666') for m in models])
        ax4.set_title('Complexity Factor by Model Type', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Complexity Factor', fontsize=10)
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, complexity in zip(bars, complexity_factors):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{complexity:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'architecture_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Architecture overview plot saved to {self.output_dir / 'architecture_overview.png'}")
    
    def plot_performance_comparison(self):
        """Plot performance comparison across models and datasets"""
        print("Generating performance comparison plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run model architecture analysis first.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different performance metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Filter out models with 0 accuracy (incompatible combinations)
        valid_df = self.df[self.df['accuracy'] > 0]
        
        if len(valid_df) == 0:
            print("No valid performance data available.")
            return
        
        # 1. Accuracy by Model and Dataset
        ax1 = axes[0, 0]
        pivot_accuracy = valid_df.pivot_table(values='accuracy', index='model_type', columns='dataset', aggfunc='mean')
        pivot_accuracy.plot(kind='bar', ax=ax1, color=[self.dataset_colors.get(d.lower(), '#666666') for d in pivot_accuracy.columns])
        ax1.set_title('Accuracy by Model and Dataset', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=10)
        ax1.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, model in enumerate(pivot_accuracy.index):
            for j, dataset in enumerate(pivot_accuracy.columns):
                value = pivot_accuracy.loc[model, dataset]
                if not pd.isna(value):
                    ax1.text(i + j*0.35 - 0.175, value + 1, f'{value:.1f}%', 
                            ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 2. F1-Score by Model and Dataset
        ax2 = axes[0, 1]
        pivot_f1 = valid_df.pivot_table(values='f1_score', index='model_type', columns='dataset', aggfunc='mean')
        pivot_f1.plot(kind='bar', ax=ax2, color=[self.dataset_colors.get(d.lower(), '#666666') for d in pivot_f1.columns])
        ax2.set_title('F1-Score by Model and Dataset', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F1-Score', fontsize=10)
        ax2.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, model in enumerate(pivot_f1.index):
            for j, dataset in enumerate(pivot_f1.columns):
                value = pivot_f1.loc[model, dataset]
                if not pd.isna(value):
                    ax2.text(i + j*0.35 - 0.175, value + 0.01, f'{value:.3f}', 
                            ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 3. Training Time by Model and Dataset
        ax3 = axes[1, 0]
        pivot_time = valid_df.pivot_table(values='training_time_factor', index='model_type', columns='dataset', aggfunc='mean')
        pivot_time.plot(kind='bar', ax=ax3, color=[self.dataset_colors.get(d.lower(), '#666666') for d in pivot_time.columns])
        ax3.set_title('Training Time Factor by Model and Dataset', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Training Time Factor (x)', fontsize=10)
        ax3.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, model in enumerate(pivot_time.index):
            for j, dataset in enumerate(pivot_time.columns):
                value = pivot_time.loc[model, dataset]
                if not pd.isna(value):
                    ax3.text(i + j*0.35 - 0.175, value + 5, f'{value:.0f}x', 
                            ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 4. Test Loss by Model and Dataset
        ax4 = axes[1, 1]
        pivot_loss = valid_df.pivot_table(values='test_loss', index='model_type', columns='dataset', aggfunc='mean')
        pivot_loss.plot(kind='bar', ax=ax4, color=[self.dataset_colors.get(d.lower(), '#666666') for d in pivot_loss.columns])
        ax4.set_title('Test Loss by Model and Dataset', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Test Loss', fontsize=10)
        ax4.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, model in enumerate(pivot_loss.index):
            for j, dataset in enumerate(pivot_loss.columns):
                value = pivot_loss.loc[model, dataset]
                if not pd.isna(value):
                    ax4.text(i + j*0.35 - 0.175, value + 0.01, f'{value:.4f}', 
                            ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance comparison plot saved to {self.output_dir / 'performance_comparison.png'}")
    
    def plot_parameter_efficiency_analysis(self):
        """Plot parameter efficiency analysis"""
        print("Generating parameter efficiency analysis plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run model architecture analysis first.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different efficiency metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Parameter Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # Filter valid data
        valid_df = self.df[self.df['accuracy'] > 0]
        
        if len(valid_df) == 0:
            print("No valid efficiency data available.")
            return
        
        # 1. Accuracy per Parameter
        ax1 = axes[0, 0]
        models = valid_df['model_type']
        accuracy_per_param = valid_df['accuracy_per_parameter']
        
        bars = ax1.bar(models, accuracy_per_param, 
                       color=[self.model_colors.get(m.lower(), '#666666') for m in models])
        ax1.set_title('Accuracy per Parameter', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy per Million Parameters', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, accuracy_per_param):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Accuracy per Training Time
        ax2 = axes[0, 1]
        accuracy_per_time = valid_df['accuracy_per_time']
        
        bars = ax2.bar(models, accuracy_per_time, 
                       color=[self.model_colors.get(m.lower(), '#666666') for m in models])
        ax2.set_title('Accuracy per Training Time', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy per Training Time Unit', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, accuracy_per_time):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Memory Efficiency
        ax3 = axes[1, 0]
        memory_efficiency = valid_df['memory_efficiency']
        
        bars = ax3.bar(models, memory_efficiency, 
                       color=[self.model_colors.get(m.lower(), '#666666') for m in models])
        ax3.set_title('Memory Efficiency', fontsize=12, fontweight='bold')
        ax3.set_ylabel('GB per Million Parameters', fontsize=10)
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, memory_efficiency):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Parameter Count vs Accuracy
        ax4 = axes[1, 1]
        parameters = valid_df['total_parameters']
        accuracies = valid_df['accuracy']
        
        # Create scatter plot with different colors for each model type
        for model_type in valid_df['model_type'].unique():
            model_data = valid_df[valid_df['model_type'] == model_type]
            ax4.scatter(model_data['total_parameters'], model_data['accuracy'],
                       label=model_type, s=100, alpha=0.7,
                       color=self.model_colors.get(model_type.lower(), '#666666'))
        
        ax4.set_title('Parameter Count vs Accuracy', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Total Parameters', fontsize=10)
        ax4.set_ylabel('Accuracy (%)', fontsize=10)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Parameter efficiency analysis plot saved to {self.output_dir / 'parameter_efficiency_analysis.png'}")
    
    def plot_architecture_type_comparison(self):
        """Plot comparison by architecture type"""
        print("Generating architecture type comparison plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run model architecture analysis first.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different architecture types
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Architecture Type Comparison', fontsize=16, fontweight='bold')
        
        # Filter valid data
        valid_df = self.df[self.df['accuracy'] > 0]
        
        if len(valid_df) == 0:
            print("No valid comparison data available.")
            return
        
        # Group by architecture type
        arch_stats = valid_df.groupby('architecture_type').agg({
            'accuracy': ['mean', 'std'],
            'total_parameters': 'mean',
            'memory_usage_gb': 'mean',
            'training_time_factor': 'mean'
        }).round(2)
        
        # 1. Average Accuracy by Architecture Type
        ax1 = axes[0, 0]
        arch_types = arch_stats.index
        avg_accuracies = arch_stats[('accuracy', 'mean')]
        std_accuracies = arch_stats[('accuracy', 'std')]
        
        bars = ax1.bar(arch_types, avg_accuracies, yerr=std_accuracies, capsize=5,
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Average Accuracy by Architecture Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=10)
        ax1.set_ylim(0, 100)
        
        # Add value labels
        for bar, mean_val, std_val in zip(bars, avg_accuracies, std_accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{mean_val:.1f}±{std_val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Average Parameters by Architecture Type
        ax2 = axes[0, 1]
        avg_parameters = arch_stats[('total_parameters', 'mean')]
        
        bars = ax2.bar(arch_types, avg_parameters,
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title('Average Parameters by Architecture Type', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Parameters', fontsize=10)
        
        # Add value labels
        for bar, param_count in zip(bars, avg_parameters):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100000,
                    f'{param_count:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Average Memory Usage by Architecture Type
        ax3 = axes[1, 0]
        avg_memory = arch_stats[('memory_usage_gb', 'mean')]
        
        bars = ax3.bar(arch_types, avg_memory,
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax3.set_title('Average Memory Usage by Architecture Type', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Memory Usage (GB)', fontsize=10)
        
        # Add value labels
        for bar, memory in zip(bars, avg_memory):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{memory:.1f} GB', ha='center', va='bottom', fontweight='bold')
        
        # 4. Average Training Time by Architecture Type
        ax4 = axes[1, 1]
        avg_training_time = arch_stats[('training_time_factor', 'mean')]
        
        bars = ax4.bar(arch_types, avg_training_time,
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax4.set_title('Average Training Time by Architecture Type', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Training Time Factor (x)', fontsize=10)
        
        # Add value labels
        for bar, time_factor in zip(bars, avg_training_time):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{time_factor:.0f}x', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'architecture_type_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Architecture type comparison plot saved to {self.output_dir / 'architecture_type_comparison.png'}")
    
    def plot_performance_heatmap(self):
        """Plot performance heatmap"""
        print("Generating performance heatmap...")
        
        if len(self.df) == 0:
            print("No data to plot. Run model architecture analysis first.")
            return
        
        # Filter valid data
        valid_df = self.df[self.df['accuracy'] > 0]
        
        if len(valid_df) == 0:
            print("No valid performance data available for heatmap.")
            return
        
        # Create heatmap data
        heatmap_data = valid_df.pivot_table(
            values='accuracy', 
            index='model_type', 
            columns='dataset', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Accuracy (%)'})
        plt.title('Model Performance Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('Model Type', fontsize=12)
        
        plt.savefig(self.output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance heatmap saved to {self.output_dir / 'performance_heatmap.png'}")
    
    def plot_efficiency_radar(self):
        """Plot efficiency radar chart"""
        print("Generating efficiency radar chart...")
        
        if len(self.df) == 0:
            print("No data to plot. Run model architecture analysis first.")
            return
        
        # Filter valid data
        valid_df = self.df[self.df['accuracy'] > 0]
        
        if len(valid_df) == 0:
            print("No valid efficiency data available.")
            return
        
        # Calculate efficiency metrics (normalized to 0-1)
        efficiency_metrics = valid_df.groupby('model_type').agg({
            'accuracy': 'mean',
            'accuracy_per_parameter': 'mean',
            'accuracy_per_time': 'mean',
            'memory_efficiency': 'mean'
        }).round(3)
        
        # Normalize metrics to 0-1 scale
        for col in efficiency_metrics.columns:
            max_val = efficiency_metrics[col].max()
            min_val = efficiency_metrics[col].min()
            if max_val > min_val:
                efficiency_metrics[col] = (efficiency_metrics[col] - min_val) / (max_val - min_val)
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # Define angles for each metric
        angles = np.linspace(0, 2 * np.pi, len(efficiency_metrics.columns), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each model
        for i, model_type in enumerate(efficiency_metrics.index):
            values = efficiency_metrics.loc[model_type].values.tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_type, 
                   color=self.model_colors.get(model_type.lower(), '#666666'))
            ax.fill(angles, values, alpha=0.25, 
                   color=self.model_colors.get(model_type.lower(), '#666666'))
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Accuracy', 'Parameter Efficiency', 'Time Efficiency', 'Memory Efficiency'])
        ax.set_ylim(0, 1)
        ax.set_title('Model Efficiency Radar Chart', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Efficiency radar chart saved to {self.output_dir / 'efficiency_radar.png'}")
    
    def plot_summary_statistics(self):
        """Plot summary statistics"""
        print("Generating summary statistics plot...")
        
        if 'summary_statistics' not in self.results:
            print("No summary statistics available.")
            return
        
        summary = self.results['summary_statistics']
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different summary metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Architecture Summary Statistics', fontsize=16, fontweight='bold')
        
        # 1. Overall statistics
        ax1 = axes[0, 0]
        metrics = ['total_models_analyzed', 'average_accuracy', 'average_training_time_factor', 'average_memory_usage_gb']
        labels = ['Total Models', 'Avg Accuracy (%)', 'Avg Training Time (x)', 'Avg Memory (GB)']
        values = [summary.get(metric, 0) for metric in metrics]
        
        bars = ax1.bar(labels, values, color=['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax1.set_title('Overall Statistics', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Value', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Best and worst performers
        ax2 = axes[0, 1]
        if summary.get('best_performing_model'):
            best_model = summary['best_performing_model']
            best_label = f"{best_model['model_type']}\n{best_model['dataset']}"
            best_accuracy = best_model['accuracy']
            
            ax2.bar(['Best Performer'], [best_accuracy], color='#4ECDC4')
            ax2.text(0, best_accuracy + 1, f'{best_accuracy:.2f}%', 
                    ha='center', va='bottom', fontweight='bold')
        
        if summary.get('worst_performing_model'):
            worst_model = summary['worst_performing_model']
            worst_label = f"{worst_model['model_type']}\n{worst_model['dataset']}"
            worst_accuracy = worst_model['accuracy']
            
            ax2.bar(['Worst Performer'], [worst_accuracy], color='#FF6B6B')
            ax2.text(1, worst_accuracy + 1, f'{worst_accuracy:.2f}%', 
                    ha='center', va='bottom', fontweight='bold')
        
        ax2.set_title('Best vs Worst Performers', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontsize=10)
        ax2.set_ylim(0, 100)
        
        # 3. Architecture type statistics
        ax3 = axes[1, 0]
        if 'architecture_statistics' in summary:
            arch_stats = summary['architecture_statistics']
            arch_types = list(arch_stats.keys())
            avg_accuracies = [arch_stats[arch]['avg_accuracy'] for arch in arch_types]
            
            bars = ax3.bar(arch_types, avg_accuracies, 
                          color=['#4ECDC4', '#45B7D1', '#96CEB4'])
            ax3.set_title('Average Accuracy by Architecture Type', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Accuracy (%)', fontsize=10)
            ax3.set_ylim(0, 100)
            
            # Add value labels
            for bar, value in zip(bars, avg_accuracies):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Parameter efficiency
        ax4 = axes[1, 1]
        if summary.get('most_efficient_model') and summary.get('least_efficient_model'):
            most_eff = summary['most_efficient_model']
            least_eff = summary['least_efficient_model']
            
            models = [f"{most_eff['model_type']}\n{most_eff['dataset']}", 
                     f"{least_eff['model_type']}\n{least_eff['dataset']}"]
            parameters = [most_eff['total_parameters'], least_eff['total_parameters']]
            
            bars = ax4.bar(models, parameters, color=['#4ECDC4', '#FF6B6B'])
            ax4.set_title('Most vs Least Efficient Models', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Total Parameters', fontsize=10)
            
            # Add value labels
            for bar, param_count in zip(bars, parameters):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000,
                        f'{param_count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary statistics plot saved to {self.output_dir / 'summary_statistics.png'}")
    
    def generate_all_plots(self):
        """Generate all model architecture plots"""
        print("Generating all model architecture visualization plots...")
        
        # Generate all plots
        self.plot_architecture_overview()
        self.plot_performance_comparison()
        self.plot_parameter_efficiency_analysis()
        self.plot_architecture_type_comparison()
        self.plot_performance_heatmap()
        self.plot_efficiency_radar()
        self.plot_summary_statistics()
        
        print(f"\nAll plots generated successfully in {self.output_dir}")
        print("Generated plots:")
        plot_files = [
            'architecture_overview.png',
            'performance_comparison.png',
            'parameter_efficiency_analysis.png',
            'architecture_type_comparison.png',
            'performance_heatmap.png',
            'efficiency_radar.png',
            'summary_statistics.png'
        ]
        
        for plot_file in plot_files:
            if (self.output_dir / plot_file).exists():
                print(f"  ✅ {plot_file}")
            else:
                print(f"  ❌ {plot_file} (no data available)")

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Generate Model Architecture Visualization Plots')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Input model architecture results JSON file')
    parser.add_argument('--output_dir', type=str, default='./architecture_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create plotter
    plotter = ModelArchitecturePlotter(args.input_file, args.output_dir)
    
    # Generate all plots
    plotter.generate_all_plots()
    
    print(f"\nModel architecture plots generation completed!")
    print(f"Output directory: {args.output_dir}")

if __name__ == '__main__':
    main() 