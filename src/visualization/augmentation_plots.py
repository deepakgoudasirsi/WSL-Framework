#!/usr/bin/env python3
"""
Augmentation Plots Generator
Generates comprehensive plots from data augmentation analysis results
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

class AugmentationPlotter:
    """Generate comprehensive plots from augmentation analysis results"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Define color scheme for augmentations
        self.augmentation_colors = {
            'random_rotation': '#FF6B6B',
            'horizontal_flip': '#4ECDC4',
            'random_crop': '#45B7D1',
            'color_jitter': '#96CEB4',
            'gaussian_noise': '#FFEAA7',
            'vertical_flip': '#DDA0DD',
            'random_shift': '#98D8C8',
            'all_combined': '#F7DC6F'
        }
        
        # Load augmentation analysis results if available
        self.augmentation_results = self._load_augmentation_results()
    
    def _load_augmentation_results(self) -> Dict[str, Any]:
        """Load augmentation analysis results from JSON files"""
        results = {}
        
        # Try to load the augmentation analysis results
        augmentation_file = Path('augmentation_analysis_results.json')
        if augmentation_file.exists():
            try:
                with open(augmentation_file, 'r') as f:
                    results = json.load(f)
                print(f"Loaded augmentation results from {augmentation_file}")
            except Exception as e:
                print(f"Warning: Could not load augmentation results: {e}")
        
        return results
    
    def plot_augmentation_performance_comparison(self):
        """Plot performance comparison across different augmentations"""
        print("Generating augmentation performance comparison plot...")
        
        if not self.augmentation_results or 'augmentation_results' not in self.augmentation_results:
            print("No augmentation results available. Run data augmentation analysis first.")
            return
        
        df = pd.DataFrame(self.augmentation_results['augmentation_results'])
        
        if len(df) == 0:
            print("No augmentation data available for plotting.")
            return
        
        plt.figure(figsize=(14, 8))
        
        # Create grouped bar plot for accuracy improvements
        augmentations = []
        accuracy_improvements = []
        colors = []
        
        for _, row in df.iterrows():
            aug_name = '+'.join(row['augmentations']) if len(row['augmentations']) > 1 else row['augmentations'][0]
            augmentations.append(aug_name)
            accuracy_improvements.append(row['accuracy_improvement'] * 100)  # Convert to percentage
            colors.append(self.augmentation_colors.get(row['augmentations'][0], '#666666'))
        
        bars = plt.bar(augmentations, accuracy_improvements, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, accuracy_improvements):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'+{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Data Augmentation Performance Impact', fontsize=16, fontweight='bold')
        plt.xlabel('Augmentation Technique', fontsize=12)
        plt.ylabel('Accuracy Improvement (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, max(accuracy_improvements) * 1.2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'augmentation_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Augmentation performance comparison plot saved to {self.output_dir / 'augmentation_performance_comparison.png'}")
    
    def plot_augmentation_efficiency_analysis(self):
        """Plot efficiency analysis of augmentations"""
        print("Generating augmentation efficiency analysis plot...")
        
        if not self.augmentation_results or 'augmentation_results' not in self.augmentation_results:
            print("No augmentation results available. Run data augmentation analysis first.")
            return
        
        df = pd.DataFrame(self.augmentation_results['augmentation_results'])
        
        if len(df) == 0:
            print("No augmentation data available for plotting.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different efficiency metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Augmentation Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # 1. Accuracy Improvement vs Training Time Increase
        ax1 = axes[0, 0]
        for _, row in df.iterrows():
            aug_name = '+'.join(row['augmentations']) if len(row['augmentations']) > 1 else row['augmentations'][0]
            ax1.scatter(row['time_increase'], row['accuracy_improvement'] * 100,
                       label=aug_name, s=100, alpha=0.7)
        ax1.set_xlabel('Training Time Increase (minutes)', fontsize=10)
        ax1.set_ylabel('Accuracy Improvement (%)', fontsize=10)
        ax1.set_title('Accuracy vs Training Time Trade-off', fontsize=12, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy Improvement vs Memory Increase
        ax2 = axes[0, 1]
        for _, row in df.iterrows():
            aug_name = '+'.join(row['augmentations']) if len(row['augmentations']) > 1 else row['augmentations'][0]
            ax2.scatter(row['memory_increase'], row['accuracy_improvement'] * 100,
                       label=aug_name, s=100, alpha=0.7)
        ax2.set_xlabel('Memory Increase (GB)', fontsize=10)
        ax2.set_ylabel('Accuracy Improvement (%)', fontsize=10)
        ax2.set_title('Accuracy vs Memory Trade-off', fontsize=12, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Training Time Increase by Augmentation
        ax3 = axes[1, 0]
        time_data = []
        time_labels = []
        for _, row in df.iterrows():
            aug_name = '+'.join(row['augmentations']) if len(row['augmentations']) > 1 else row['augmentations'][0]
            time_data.append(row['time_increase'])
            time_labels.append(aug_name)
        
        bars = ax3.bar(time_labels, time_data, color=[self.augmentation_colors.get(label.split('+')[0], '#666666') for label in time_labels])
        ax3.set_xlabel('Augmentation Technique', fontsize=10)
        ax3.set_ylabel('Training Time Increase (minutes)', fontsize=10)
        ax3.set_title('Training Time Impact by Augmentation', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, time_data):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'+{value:.1f}min', ha='center', va='bottom', fontweight='bold')
        
        # 4. Memory Increase by Augmentation
        ax4 = axes[1, 1]
        memory_data = []
        memory_labels = []
        for _, row in df.iterrows():
            aug_name = '+'.join(row['augmentations']) if len(row['augmentations']) > 1 else row['augmentations'][0]
            memory_data.append(row['memory_increase'])
            memory_labels.append(aug_name)
        
        bars = ax4.bar(memory_labels, memory_data, color=[self.augmentation_colors.get(label.split('+')[0], '#666666') for label in memory_labels])
        ax4.set_xlabel('Augmentation Technique', fontsize=10)
        ax4.set_ylabel('Memory Increase (GB)', fontsize=10)
        ax4.set_title('Memory Impact by Augmentation', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, memory_data):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'+{value:.1f}GB', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'augmentation_efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Augmentation efficiency analysis plot saved to {self.output_dir / 'augmentation_efficiency_analysis.png'}")
    
    def plot_augmentation_heatmap(self):
        """Plot heatmap of augmentation performance metrics"""
        print("Generating augmentation performance heatmap...")
        
        if not self.augmentation_results or 'augmentation_results' not in self.augmentation_results:
            print("No augmentation results available. Run data augmentation analysis first.")
            return
        
        df = pd.DataFrame(self.augmentation_results['augmentation_results'])
        
        if len(df) == 0:
            print("No augmentation data available for plotting.")
            return
        
        # Prepare data for heatmap
        heatmap_data = []
        row_labels = []
        
        for _, row in df.iterrows():
            aug_name = '+'.join(row['augmentations']) if len(row['augmentations']) > 1 else row['augmentations'][0]
            row_labels.append(aug_name)
            
            # Normalize metrics for heatmap
            accuracy_norm = row['accuracy_improvement'] * 100  # Convert to percentage
            time_norm = row['time_increase'] / 10  # Normalize time (assuming max 10 min increase)
            memory_norm = row['memory_increase'] / 0.5  # Normalize memory (assuming max 0.5 GB increase)
            
            heatmap_data.append([accuracy_norm, time_norm, memory_norm])
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                index=row_labels,
                                columns=['Accuracy\nImprovement (%)', 'Time\nIncrease (norm)', 'Memory\nIncrease (norm)'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Normalized Metric Value'})
        plt.title('Data Augmentation Performance Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Performance Metrics', fontsize=12)
        plt.ylabel('Augmentation Techniques', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'augmentation_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Augmentation performance heatmap saved to {self.output_dir / 'augmentation_performance_heatmap.png'}")
    
    def plot_augmentation_summary_statistics(self):
        """Plot summary statistics from augmentation analysis"""
        print("Generating augmentation summary statistics plot...")
        
        if not self.augmentation_results or 'summary_statistics' not in self.augmentation_results:
            print("No summary statistics available.")
            return
        
        summary = self.augmentation_results['summary_statistics']
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different summary metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Augmentation Summary Statistics', fontsize=16, fontweight='bold')
        
        # 1. Overall statistics
        if 'overall' in summary:
            overall = summary['overall']
            ax1 = axes[0, 0]
            metrics = ['avg_accuracy_improvement', 'avg_time_increase', 'avg_memory_increase']
            labels = ['Accuracy\nImprovement (%)', 'Time\nIncrease (min)', 'Memory\nIncrease (GB)']
            values = [overall.get(metric, 0) * 100 if 'accuracy' in metric else overall.get(metric, 0) for metric in metrics]
            
            bars = ax1.bar(labels, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax1.set_title('Overall Performance Metrics', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Value', fontsize=10)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Individual augmentation results
        if 'by_augmentation' in summary:
            ax2 = axes[0, 1]
            aug_data = summary['by_augmentation']
            
            if aug_data:
                aug_names = list(aug_data.keys())
                accuracy_improvements = [aug_data[aug]['accuracy_improvement'] * 100 for aug in aug_names]
                time_increases = [aug_data[aug]['time_increase'] for aug in aug_names]
                
                x = np.arange(len(aug_names))
                width = 0.35
                
                bars1 = ax2.bar(x - width/2, accuracy_improvements, width, label='Accuracy Improvement (%)', color='#FF6B6B')
                ax2_twin = ax2.twinx()
                bars2 = ax2_twin.bar(x + width/2, time_increases, width, label='Time Increase (min)', color='#4ECDC4')
                
                ax2.set_xlabel('Augmentation Technique', fontsize=10)
                ax2.set_ylabel('Accuracy Improvement (%)', fontsize=10, color='#FF6B6B')
                ax2_twin.set_ylabel('Time Increase (min)', fontsize=10, color='#4ECDC4')
                ax2.set_title('Individual Augmentation Performance', fontsize=12, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(aug_names, rotation=45)
                ax2.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars1, accuracy_improvements):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                for bar, value in zip(bars2, time_increases):
                    ax2_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                 f'{value:.1f}min', ha='center', va='bottom', fontweight='bold')
        
        # 3. Performance rankings
        if 'performance_analysis' in summary:
            perf_analysis = summary['performance_analysis']
            ax3 = axes[1, 0]
            
            if 'best_performing' in perf_analysis:
                best = perf_analysis['best_performing']
                ax3.text(0.5, 0.7, f"Best Overall:\n{best['augmentations']}\n{best['final_accuracy']:.3f} accuracy", 
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FF6B6B", alpha=0.7))
            
            if 'most_efficient' in perf_analysis:
                most_eff = perf_analysis['most_efficient']
                ax3.text(0.5, 0.3, f"Most Efficient:\n{most_eff['augmentations']}\n{most_eff['final_training_time']:.1f} min", 
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#4ECDC4", alpha=0.7))
            
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.set_title('Performance Rankings', fontsize=12, fontweight='bold')
            ax3.axis('off')
        
        # 4. Dataset and model information
        if 'metadata' in self.augmentation_results:
            metadata = self.augmentation_results['metadata']
            ax4 = axes[1, 1]
            
            info_text = f"Dataset: {metadata['dataset'].upper()}\n"
            info_text += f"Model Type: {metadata['model_type'].title()}\n"
            info_text += f"Valid Augmentations: {len(metadata['valid_augmentations'])}\n"
            info_text += f"Training Epochs: {metadata['epochs']}\n"
            info_text += f"Total Combinations: {metadata['total_combinations']}"
            
            ax4.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#96CEB4", alpha=0.7))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.set_title('Analysis Configuration', fontsize=12, fontweight='bold')
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'augmentation_summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Augmentation summary statistics plot saved to {self.output_dir / 'augmentation_summary_statistics.png'}")
    
    def plot_augmentation_recommendations(self):
        """Plot augmentation recommendations based on analysis"""
        print("Generating augmentation recommendations plot...")
        
        if not self.augmentation_results:
            print("No augmentation results available. Run data augmentation analysis first.")
            return
        
        # Create recommendations based on analysis
        recommendations = self._generate_recommendations()
        
        plt.figure(figsize=(12, 8))
        
        # Create a recommendation chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define recommendation categories
        categories = ['Best Performance', 'Most Efficient', 'Balanced Approach', 'Production Ready']
        scores = [recommendations.get('best_performance', 0), 
                 recommendations.get('most_efficient', 0),
                 recommendations.get('balanced', 0),
                 recommendations.get('production', 0)]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = ax.bar(categories, scores, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Data Augmentation Recommendations', fontsize=16, fontweight='bold')
        ax.set_ylabel('Recommendation Score', fontsize=12)
        ax.set_ylim(0, max(scores) * 1.2)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'augmentation_recommendations.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Augmentation recommendations plot saved to {self.output_dir / 'augmentation_recommendations.png'}")
    
    def _generate_recommendations(self) -> Dict[str, float]:
        """Generate recommendations based on augmentation analysis"""
        recommendations = {
            'best_performance': 0.0,
            'most_efficient': 0.0,
            'balanced': 0.0,
            'production': 0.0
        }
        
        if not self.augmentation_results or 'augmentation_results' not in self.augmentation_results:
            return recommendations
        
        df = pd.DataFrame(self.augmentation_results['augmentation_results'])
        
        if len(df) == 0:
            return recommendations
        
        # Calculate recommendation scores based on analysis
        max_accuracy_improvement = df['accuracy_improvement'].max()
        min_time_increase = df['time_increase'].min()
        min_memory_increase = df['memory_increase'].min()
        
        # Best performance score
        best_perf_row = df.loc[df['accuracy_improvement'].idxmax()]
        recommendations['best_performance'] = best_perf_row['accuracy_improvement'] * 100
        
        # Most efficient score
        most_eff_row = df.loc[df['time_increase'].idxmin()]
        efficiency_score = (1 / (1 + most_eff_row['time_increase'])) * 100
        recommendations['most_efficient'] = efficiency_score
        
        # Balanced approach score
        balanced_score = (df['accuracy_improvement'].mean() * 100 + 
                         (1 / (1 + df['time_increase'].mean())) * 50) / 2
        recommendations['balanced'] = balanced_score
        
        # Production ready score
        production_score = (df['accuracy_improvement'].mean() * 100 + 
                          (1 / (1 + df['memory_increase'].mean())) * 50) / 2
        recommendations['production'] = production_score
        
        return recommendations
    
    def generate_all_plots(self):
        """Generate all augmentation visualization plots"""
        print("Generating all augmentation visualization plots...")
        
        # Generate all plots
        self.plot_augmentation_performance_comparison()
        self.plot_augmentation_efficiency_analysis()
        self.plot_augmentation_heatmap()
        self.plot_augmentation_summary_statistics()
        self.plot_augmentation_recommendations()
        
        print(f"\nAll plots generated successfully in {self.output_dir}")
        print("Generated plots:")
        plot_files = [
            'augmentation_performance_comparison.png',
            'augmentation_efficiency_analysis.png',
            'augmentation_performance_heatmap.png',
            'augmentation_summary_statistics.png',
            'augmentation_recommendations.png'
        ]
        
        for plot_file in plot_files:
            if (self.output_dir / plot_file).exists():
                print(f"  ✅ {plot_file}")
            else:
                print(f"  ❌ {plot_file} (no data available)")

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Generate Augmentation Visualization Plots')
    parser.add_argument('--output_dir', type=str, default='./augmentation_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create plotter
    plotter = AugmentationPlotter(args.output_dir)
    
    # Generate all plots
    plotter.generate_all_plots()
    
    print(f"\nAugmentation plots generation completed!")
    print(f"Output directory: {args.output_dir}")

if __name__ == '__main__':
    main() 