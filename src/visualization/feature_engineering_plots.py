#!/usr/bin/env python3
"""
Feature Engineering Plots Generator
Generates comprehensive plots from feature engineering analysis results
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

class FeatureEngineeringPlotter:
    """Generate comprehensive plots from feature engineering analysis results"""
    
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load feature engineering results
        with open(self.input_file, 'r') as f:
            self.results = json.load(f)
        
        # Convert to DataFrame for easier plotting
        if 'feature_engineering_results' in self.results:
            self.df = pd.DataFrame(self.results['feature_engineering_results'])
        else:
            self.df = pd.DataFrame()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Define color scheme for strategies
        self.strategy_colors = {
            'consistency': '#4ECDC4',
            'pseudo_label': '#45B7D1',
            'co_training': '#96CEB4',
            'combined': '#FFEAA7'
        }
    
    def plot_quality_comparison(self):
        """Plot quality score comparison across strategies"""
        print("Generating quality comparison plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run feature engineering analysis first.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Group by strategy and calculate average quality metrics
        strategy_stats = self.df.groupby('strategy').agg({
            'quality_score': ['mean', 'std'],
            'feature_completeness': ['mean', 'std'],
            'feature_relevance': ['mean', 'std'],
            'feature_diversity': ['mean', 'std']
        }).round(3)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Feature Engineering Quality Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['quality_score', 'feature_completeness', 'feature_relevance', 'feature_diversity']
        titles = ['Quality Score', 'Feature Completeness', 'Feature Relevance', 'Feature Diversity']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            # Get data for this metric
            means = strategy_stats[metric]['mean']
            stds = strategy_stats[metric]['std']
            
            # Create bar plot
            bars = ax.bar(means.index, means.values, yerr=stds.values, 
                         capsize=5, color=[self.strategy_colors.get(s.lower(), '#666666') for s in means.index])
            
            # Add value labels
            for bar, mean_val, std_val in zip(bars, means.values, stds.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{mean_val:.3f}±{std_val:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Quality comparison plot saved to {self.output_dir / 'quality_comparison.png'}")
    
    def plot_efficiency_analysis(self):
        """Plot computational efficiency analysis"""
        print("Generating efficiency analysis plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run feature engineering analysis first.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different efficiency metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Feature Engineering Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # 1. Extraction Time vs Quality Score
        ax1 = axes[0, 0]
        for strategy in self.df['strategy'].unique():
            strategy_data = self.df[self.df['strategy'] == strategy]
            ax1.scatter(strategy_data['extraction_time'], strategy_data['quality_score'],
                       label=strategy.title(), s=100, alpha=0.7)
        ax1.set_xlabel('Extraction Time (seconds)', fontsize=10)
        ax1.set_ylabel('Quality Score', fontsize=10)
        ax1.set_title('Extraction Time vs Quality Score', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Memory Usage vs Quality Score
        ax2 = axes[0, 1]
        for strategy in self.df['strategy'].unique():
            strategy_data = self.df[self.df['strategy'] == strategy]
            ax2.scatter(strategy_data['memory_usage'], strategy_data['quality_score'],
                       label=strategy.title(), s=100, alpha=0.7)
        ax2.set_xlabel('Memory Usage (MB)', fontsize=10)
        ax2.set_ylabel('Quality Score', fontsize=10)
        ax2.set_title('Memory Usage vs Quality Score', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Computational Efficiency by Strategy
        ax3 = axes[1, 0]
        efficiency_stats = self.df.groupby('strategy')['computational_efficiency'].agg(['mean', 'std']).round(3)
        bars = ax3.bar(efficiency_stats.index, efficiency_stats['mean'], 
                       yerr=efficiency_stats['std'], capsize=5,
                       color=[self.strategy_colors.get(s.lower(), '#666666') for s in efficiency_stats.index])
        
        for bar, mean_val, std_val in zip(bars, efficiency_stats['mean'], efficiency_stats['std']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean_val:.3f}±{std_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_title('Computational Efficiency by Strategy', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Efficiency Score', fontsize=10)
        ax3.set_ylim(0, 1.1)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Training Time vs Robustness Score
        ax4 = axes[1, 1]
        for strategy in self.df['strategy'].unique():
            strategy_data = self.df[self.df['strategy'] == strategy]
            ax4.scatter(strategy_data['training_time'], strategy_data['robustness_score'],
                       label=strategy.title(), s=100, alpha=0.7)
        ax4.set_xlabel('Training Time (minutes)', fontsize=10)
        ax4.set_ylabel('Robustness Score', fontsize=10)
        ax4.set_title('Training Time vs Robustness Score', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Efficiency analysis plot saved to {self.output_dir / 'efficiency_analysis.png'}")
    
    def plot_strategy_comparison(self):
        """Plot comprehensive strategy comparison"""
        print("Generating strategy comparison plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run feature engineering analysis first.")
            return
        
        # Calculate average metrics by strategy
        strategy_metrics = self.df.groupby('strategy').agg({
            'quality_score': 'mean',
            'extraction_time': 'mean',
            'memory_usage': 'mean',
            'computational_efficiency': 'mean',
            'training_time': 'mean',
            'robustness_score': 'mean'
        }).round(3)
        
        # Normalize metrics for radar chart (0-1 scale)
        normalized_metrics = strategy_metrics.copy()
        for col in normalized_metrics.columns:
            if col != 'quality_score' and col != 'computational_efficiency' and col != 'robustness_score':
                # For time and memory, lower is better, so invert
                normalized_metrics[col] = 1 - (normalized_metrics[col] - normalized_metrics[col].min()) / (normalized_metrics[col].max() - normalized_metrics[col].min())
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
        
        # Define angles for each metric
        metrics = ['quality_score', 'computational_efficiency', 'robustness_score', 
                  'extraction_time', 'memory_usage', 'training_time']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each strategy
        for strategy in normalized_metrics.index:
            values = normalized_metrics.loc[strategy, metrics].values.tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=strategy.title(), 
                   color=self.strategy_colors.get(strategy.lower(), '#666666'))
            ax.fill(angles, values, alpha=0.25, color=self.strategy_colors.get(strategy.lower(), '#666666'))
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Strategy Performance Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'strategy_comparison_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Strategy comparison radar plot saved to {self.output_dir / 'strategy_comparison_radar.png'}")
    
    def plot_dataset_analysis(self):
        """Plot dataset-specific analysis"""
        print("Generating dataset analysis plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run feature engineering analysis first.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for dataset analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Feature Engineering Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Quality Score by Dataset and Strategy
        ax1 = axes[0, 0]
        pivot_quality = self.df.pivot_table(values='quality_score', index='dataset', columns='strategy', aggfunc='mean')
        pivot_quality.plot(kind='bar', ax=ax1, color=[self.strategy_colors.get(s.lower(), '#666666') for s in pivot_quality.columns])
        ax1.set_title('Quality Score by Dataset and Strategy', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Quality Score', fontsize=10)
        ax1.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Extraction Time by Dataset and Strategy
        ax2 = axes[0, 1]
        pivot_time = self.df.pivot_table(values='extraction_time', index='dataset', columns='strategy', aggfunc='mean')
        pivot_time.plot(kind='bar', ax=ax2, color=[self.strategy_colors.get(s.lower(), '#666666') for s in pivot_time.columns])
        ax2.set_title('Extraction Time by Dataset and Strategy', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Extraction Time (seconds)', fontsize=10)
        ax2.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Memory Usage by Dataset and Strategy
        ax3 = axes[1, 0]
        pivot_memory = self.df.pivot_table(values='memory_usage', index='dataset', columns='strategy', aggfunc='mean')
        pivot_memory.plot(kind='bar', ax=ax3, color=[self.strategy_colors.get(s.lower(), '#666666') for s in pivot_memory.columns])
        ax3.set_title('Memory Usage by Dataset and Strategy', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Memory Usage (MB)', fontsize=10)
        ax3.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Training Time by Dataset and Strategy
        ax4 = axes[1, 1]
        pivot_training = self.df.pivot_table(values='training_time', index='dataset', columns='strategy', aggfunc='mean')
        pivot_training.plot(kind='bar', ax=ax4, color=[self.strategy_colors.get(s.lower(), '#666666') for s in pivot_training.columns])
        ax4.set_title('Training Time by Dataset and Strategy', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Training Time (minutes)', fontsize=10)
        ax4.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Dataset analysis plot saved to {self.output_dir / 'dataset_analysis.png'}")
    
    def plot_performance_heatmap(self):
        """Plot performance heatmap"""
        print("Generating performance heatmap...")
        
        if len(self.df) == 0:
            print("No data to plot. Run feature engineering analysis first.")
            return
        
        # Create heatmap data
        heatmap_data = self.df.pivot_table(
            values='quality_score', 
            index='dataset', 
            columns='strategy', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Quality Score'})
        plt.title('Feature Engineering Quality Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Strategy', fontsize=12)
        plt.ylabel('Dataset', fontsize=12)
        
        plt.savefig(self.output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance heatmap saved to {self.output_dir / 'performance_heatmap.png'}")
    
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
        fig.suptitle('Feature Engineering Summary Statistics', fontsize=16, fontweight='bold')
        
        # 1. Overall statistics
        if 'overall' in summary:
            overall = summary['overall']
            ax1 = axes[0, 0]
            metrics = ['avg_quality_score', 'avg_extraction_time', 'avg_memory_usage', 'avg_computational_efficiency']
            labels = ['Quality Score', 'Extraction Time (s)', 'Memory Usage (MB)', 'Computational Efficiency']
            values = [overall.get(metric, 0) for metric in metrics]
            
            bars = ax1.bar(labels, values, color=['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
            ax1.set_title('Overall Performance Metrics', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Value', fontsize=10)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Strategy comparison
        if 'by_strategy' in summary:
            ax2 = axes[0, 1]
            strategies = list(summary['by_strategy'].keys())
            quality_scores = [summary['by_strategy'][s]['avg_quality_score'] for s in strategies]
            
            bars = ax2.bar(strategies, quality_scores, 
                          color=[self.strategy_colors.get(s.lower(), '#666666') for s in strategies])
            ax2.set_title('Average Quality Score by Strategy', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Quality Score', fontsize=10)
            ax2.set_ylim(0, 1.1)
            
            # Add value labels
            for bar, value in zip(bars, quality_scores):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Dataset comparison
        if 'by_dataset' in summary:
            ax3 = axes[1, 0]
            datasets = list(summary['by_dataset'].keys())
            quality_scores = [summary['by_dataset'][d]['avg_quality_score'] for d in datasets]
            
            bars = ax3.bar(datasets, quality_scores, color=['#4ECDC4', '#45B7D1'])
            ax3.set_title('Average Quality Score by Dataset', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Quality Score', fontsize=10)
            ax3.set_ylim(0, 1.1)
            
            # Add value labels
            for bar, value in zip(bars, quality_scores):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Performance rankings
        if 'performance_rankings' in self.results and 'by_quality_score' in self.results['performance_rankings']:
            ax4 = axes[1, 1]
            rankings = self.results['performance_rankings']['by_quality_score'][:5]
            
            strategies = [r['strategy'] for r in rankings]
            scores = [r['quality_score'] for r in rankings]
            
            bars = ax4.barh(strategies, scores, 
                           color=[self.strategy_colors.get(s.lower(), '#666666') for s in strategies])
            ax4.set_title('Top 5 Quality Score Rankings', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Quality Score', fontsize=10)
            ax4.set_xlim(0, 1.1)
            
            # Add value labels
            for bar, value in zip(bars, scores):
                ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary statistics plot saved to {self.output_dir / 'summary_statistics.png'}")
    
    def generate_all_plots(self):
        """Generate all feature engineering plots"""
        print("Generating all feature engineering visualization plots...")
        
        # Generate all plots
        self.plot_quality_comparison()
        self.plot_efficiency_analysis()
        self.plot_strategy_comparison()
        self.plot_dataset_analysis()
        self.plot_performance_heatmap()
        self.plot_summary_statistics()
        
        print(f"\nAll plots generated successfully in {self.output_dir}")
        print("Generated plots:")
        plot_files = [
            'quality_comparison.png',
            'efficiency_analysis.png',
            'strategy_comparison_radar.png',
            'dataset_analysis.png',
            'performance_heatmap.png',
            'summary_statistics.png'
        ]
        
        for plot_file in plot_files:
            if (self.output_dir / plot_file).exists():
                print(f"  ✅ {plot_file}")
            else:
                print(f"  ❌ {plot_file} (no data available)")

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Generate Feature Engineering Visualization Plots')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Input feature engineering results JSON file')
    parser.add_argument('--output_dir', type=str, default='./feature_engineering_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create plotter
    plotter = FeatureEngineeringPlotter(args.input_file, args.output_dir)
    
    # Generate all plots
    plotter.generate_all_plots()
    
    print(f"\nFeature engineering plots generation completed!")
    print(f"Output directory: {args.output_dir}")

if __name__ == '__main__':
    main() 