#!/usr/bin/env python3
"""
Performance Visualization Generator
Generates various plots from performance comparison reports
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

class PerformanceVisualizer:
    """Generate visualizations from performance reports"""
    
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load performance report
        with open(self.input_file, 'r') as f:
            self.report = json.load(f)
        
        # Convert to DataFrame for easier plotting
        self.df = pd.DataFrame(self.report['comparison_table'])
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_accuracy_comparison(self):
        """Plot accuracy comparison across different configurations"""
        print("Generating accuracy comparison plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run experiments first.")
            return
        
        # Filter out None values
        df_plot = self.df.dropna(subset=['Accuracy'])
        
        if len(df_plot) == 0:
            print("No accuracy data available for plotting.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar plot
        if 'Strategy' in df_plot.columns and 'Model_Type' in df_plot.columns:
            # Group by strategy and model type
            pivot_data = df_plot.pivot_table(
                values='Accuracy', 
                index='Model_Type', 
                columns='Strategy', 
                aggfunc='mean'
            )
            
            ax = pivot_data.plot(kind='bar', figsize=(14, 8))
            plt.title('Accuracy Comparison by Model Type and Strategy', fontsize=16, fontweight='bold')
            plt.xlabel('Model Type', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        else:
            # Simple bar plot
            plt.bar(range(len(df_plot)), df_plot['Accuracy'])
            plt.title('Accuracy Comparison Across Experiments', fontsize=16, fontweight='bold')
            plt.xlabel('Experiment Index', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.xticks(range(len(df_plot)), [f"{row['Dataset']}-{row['Model_Type']}-{row['Strategy']}" 
                                           for _, row in df_plot.iterrows()], rotation=45)
        
        plt.savefig(self.output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Accuracy comparison plot saved to {self.output_dir / 'accuracy_comparison.png'}")
    
    def plot_dataset_performance(self):
        """Plot performance comparison across datasets"""
        print("Generating dataset performance plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run experiments first.")
            return
        
        df_plot = self.df.dropna(subset=['Accuracy'])
        
        if len(df_plot) == 0:
            print("No accuracy data available for plotting.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Group by dataset
        dataset_stats = df_plot.groupby('Dataset')['Accuracy'].agg(['mean', 'std', 'count']).reset_index()
        
        # Create bar plot with error bars
        bars = plt.bar(dataset_stats['Dataset'], dataset_stats['mean'], 
                      yerr=dataset_stats['std'], capsize=5)
        
        # Add value labels on bars
        for bar, mean_val, count in zip(bars, dataset_stats['mean'], dataset_stats['count']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{mean_val:.1f}%\n(n={count})', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title('Average Performance by Dataset', fontsize=16, fontweight='bold')
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('Average Accuracy (%)', fontsize=12)
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        
        plt.savefig(self.output_dir / 'dataset_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Dataset performance plot saved to {self.output_dir / 'dataset_performance.png'}")
    
    def plot_model_performance(self):
        """Plot performance comparison across model types"""
        print("Generating model performance plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run experiments first.")
            return
        
        df_plot = self.df.dropna(subset=['Accuracy'])
        
        if len(df_plot) == 0:
            print("No accuracy data available for plotting.")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Group by model type
        model_stats = df_plot.groupby('Model_Type')['Accuracy'].agg(['mean', 'std', 'count']).reset_index()
        
        # Create bar plot with error bars
        bars = plt.bar(model_stats['Model_Type'], model_stats['mean'], 
                      yerr=model_stats['std'], capsize=5)
        
        # Add value labels on bars
        for bar, mean_val, count in zip(bars, model_stats['mean'], model_stats['count']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{mean_val:.1f}%\n(n={count})', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title('Average Performance by Model Type', fontsize=16, fontweight='bold')
        plt.xlabel('Model Type', fontsize=12)
        plt.ylabel('Average Accuracy (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        
        plt.savefig(self.output_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Model performance plot saved to {self.output_dir / 'model_performance.png'}")
    
    def plot_strategy_performance(self):
        """Plot performance comparison across strategies"""
        print("Generating strategy performance plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run experiments first.")
            return
        
        df_plot = self.df.dropna(subset=['Accuracy'])
        
        if len(df_plot) == 0:
            print("No accuracy data available for plotting.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Group by strategy
        strategy_stats = df_plot.groupby('Strategy')['Accuracy'].agg(['mean', 'std', 'count']).reset_index()
        
        # Create bar plot with error bars
        bars = plt.bar(strategy_stats['Strategy'], strategy_stats['mean'], 
                      yerr=strategy_stats['std'], capsize=5)
        
        # Add value labels on bars
        for bar, mean_val, count in zip(bars, strategy_stats['mean'], strategy_stats['count']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{mean_val:.1f}%\n(n={count})', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title('Average Performance by Strategy', fontsize=16, fontweight='bold')
        plt.xlabel('Strategy', fontsize=12)
        plt.ylabel('Average Accuracy (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        
        plt.savefig(self.output_dir / 'strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Strategy performance plot saved to {self.output_dir / 'strategy_performance.png'}")
    
    def plot_heatmap(self):
        """Plot performance heatmap"""
        print("Generating performance heatmap...")
        
        if len(self.df) == 0:
            print("No data to plot. Run experiments first.")
            return
        
        df_plot = self.df.dropna(subset=['Accuracy'])
        
        if len(df_plot) == 0:
            print("No accuracy data available for plotting.")
            return
        
        # Create pivot table for heatmap
        if 'Strategy' in df_plot.columns and 'Model_Type' in df_plot.columns:
            heatmap_data = df_plot.pivot_table(
                values='Accuracy', 
                index='Model_Type', 
                columns='Strategy', 
                aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                       cbar_kws={'label': 'Accuracy (%)'})
            plt.title('Performance Heatmap: Model Type vs Strategy', fontsize=16, fontweight='bold')
            plt.xlabel('Strategy', fontsize=12)
            plt.ylabel('Model Type', fontsize=12)
            
            plt.savefig(self.output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Performance heatmap saved to {self.output_dir / 'performance_heatmap.png'}")
    
    def plot_training_time_analysis(self):
        """Plot training time analysis"""
        print("Generating training time analysis plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run experiments first.")
            return
        
        df_plot = self.df.dropna(subset=['Training_Time_Min'])
        
        if len(df_plot) == 0:
            print("No training time data available for plotting.")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Create scatter plot: accuracy vs training time
        plt.scatter(df_plot['Training_Time_Min'], df_plot['Accuracy'], 
                   alpha=0.7, s=100)
        
        # Add labels for each point
        for _, row in df_plot.iterrows():
            plt.annotate(f"{row['Dataset']}-{row['Model_Type']}", 
                        (row['Training_Time_Min'], row['Accuracy']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title('Accuracy vs Training Time', fontsize=16, fontweight='bold')
        plt.xlabel('Training Time (minutes)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / 'training_time_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training time analysis plot saved to {self.output_dir / 'training_time_analysis.png'}")
    
    def plot_memory_usage_analysis(self):
        """Plot memory usage analysis"""
        print("Generating memory usage analysis plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run experiments first.")
            return
        
        df_plot = self.df.dropna(subset=['Memory_Usage_GB'])
        
        if len(df_plot) == 0:
            print("No memory usage data available for plotting.")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Create scatter plot: accuracy vs memory usage
        plt.scatter(df_plot['Memory_Usage_GB'], df_plot['Accuracy'], 
                   alpha=0.7, s=100)
        
        # Add labels for each point
        for _, row in df_plot.iterrows():
            plt.annotate(f"{row['Dataset']}-{row['Model_Type']}", 
                        (row['Memory_Usage_GB'], row['Accuracy']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title('Accuracy vs Memory Usage', fontsize=16, fontweight='bold')
        plt.xlabel('Memory Usage (GB)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / 'memory_usage_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Memory usage analysis plot saved to {self.output_dir / 'memory_usage_analysis.png'}")
    
    def plot_ranking_chart(self):
        """Plot ranking chart of top performers"""
        print("Generating ranking chart...")
        
        if 'rankings' not in self.report or 'by_accuracy' not in self.report['rankings']:
            print("No ranking data available.")
            return
        
        rankings = self.report['rankings']['by_accuracy']
        
        if len(rankings) == 0:
            print("No ranking data available for plotting.")
            return
        
        # Take top 10 results
        top_results = rankings[:10]
        
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart
        y_pos = range(len(top_results))
        accuracies = [result['accuracy'] for result in top_results]
        labels = [f"{result['dataset']}-{result['model_type']}-{result['strategy']}" 
                 for result in top_results]
        
        bars = plt.barh(y_pos, accuracies)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{acc:.2f}%', ha='left', va='center', fontweight='bold')
        
        plt.yticks(y_pos, labels)
        plt.xlabel('Accuracy (%)', fontsize=12)
        plt.title('Top 10 Performance Rankings', fontsize=16, fontweight='bold')
        plt.xlim(0, 100)
        plt.grid(axis='x', alpha=0.3)
        
        plt.savefig(self.output_dir / 'performance_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance rankings plot saved to {self.output_dir / 'performance_rankings.png'}")
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("Generating all performance visualization plots...")
        
        # Generate all plots
        self.plot_accuracy_comparison()
        self.plot_dataset_performance()
        self.plot_model_performance()
        self.plot_strategy_performance()
        self.plot_heatmap()
        self.plot_training_time_analysis()
        self.plot_memory_usage_analysis()
        self.plot_ranking_chart()
        
        print(f"\nAll plots generated successfully in {self.output_dir}")
        print("Generated plots:")
        plot_files = [
            'accuracy_comparison.png',
            'dataset_performance.png',
            'model_performance.png',
            'strategy_performance.png',
            'performance_heatmap.png',
            'training_time_analysis.png',
            'memory_usage_analysis.png',
            'performance_rankings.png'
        ]
        
        for plot_file in plot_files:
            if (self.output_dir / plot_file).exists():
                print(f"  ✅ {plot_file}")
            else:
                print(f"  ❌ {plot_file} (no data available)")

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Generate Performance Visualization Plots')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Input performance report JSON file')
    parser.add_argument('--output_dir', type=str, default='./performance_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = PerformanceVisualizer(args.input_file, args.output_dir)
    
    # Generate all plots
    visualizer.generate_all_plots()

if __name__ == '__main__':
    main() 