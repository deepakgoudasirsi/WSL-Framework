#!/usr/bin/env python3
"""
Dataset Quality Plots Generator
Generates comprehensive plots from dataset quality analysis results
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

class DatasetQualityPlotter:
    """Generate comprehensive plots from dataset quality analysis results"""
    
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset quality results
        with open(self.input_file, 'r') as f:
            self.results = json.load(f)
        
        # Convert to DataFrame for easier plotting
        if 'dataset_quality_results' in self.results:
            self.df = pd.DataFrame(self.results['dataset_quality_results'])
        else:
            self.df = pd.DataFrame()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Define color scheme for datasets
        self.dataset_colors = {
            'cifar10': '#FF6B6B',
            'mnist': '#4ECDC4',
            'cifar100': '#45B7D1',
            'svhn': '#96CEB4'
        }
    
    def plot_quality_overview(self):
        """Plot dataset quality overview"""
        print("Generating quality overview plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run dataset quality analysis first.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different quality metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dataset Quality Overview', fontsize=16, fontweight='bold')
        
        # 1. Overall Quality Score
        ax1 = axes[0, 0]
        datasets = self.df['dataset']
        quality_scores = self.df['overall_quality_score']
        
        bars = ax1.bar(datasets, quality_scores, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax1.set_title('Overall Quality Score', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Quality Score', fontsize=10)
        ax1.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, quality_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Dataset Specifications
        ax2 = axes[0, 1]
        total_samples = [specs['total_samples'] for specs in self.df['specifications']]
        training_samples = [specs['training_samples'] for specs in self.df['specifications']]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, total_samples, width, label='Total Samples', alpha=0.8)
        bars2 = ax2.bar(x + width/2, training_samples, width, label='Training Samples', alpha=0.8)
        
        ax2.set_title('Dataset Size Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Samples', fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(datasets)
        ax2.legend()
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 1000,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 1000,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=8)
        
        # 3. Feature Count Comparison
        ax3 = axes[1, 0]
        feature_counts = [specs['total_features'] for specs in self.df['specifications']]
        
        bars = ax3.bar(datasets, feature_counts, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax3.set_title('Total Features Comparison', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Features', fontsize=10)
        
        # Add value labels
        for bar, count in zip(bars, feature_counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Labeled vs Unlabeled Ratio
        ax4 = axes[1, 1]
        labeled_ratios = [specs['labeled_ratio'] * 100 for specs in self.df['specifications']]
        unlabeled_ratios = [specs['unlabeled_ratio'] * 100 for specs in self.df['specifications']]
        
        bars1 = ax4.bar(x - width/2, labeled_ratios, width, label='Labeled (%)', alpha=0.8)
        bars2 = ax4.bar(x + width/2, unlabeled_ratios, width, label='Unlabeled (%)', alpha=0.8)
        
        ax4.set_title('Labeled vs Unlabeled Ratio', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Percentage (%)', fontsize=10)
        ax4.set_xticks(x)
        ax4.set_xticklabels(datasets)
        ax4.legend()
        ax4.set_ylim(0, 100)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Quality overview plot saved to {self.output_dir / 'quality_overview.png'}")
    
    def plot_quality_metrics_comparison(self):
        """Plot comparison of quality metrics across datasets"""
        print("Generating quality metrics comparison plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run dataset quality analysis first.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different quality metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dataset Quality Metrics Comparison', fontsize=16, fontweight='bold')
        
        datasets = self.df['dataset']
        
        # 1. Completeness Metrics
        ax1 = axes[0, 0]
        completeness_scores = [analysis['overall_completeness'] for analysis in self.df['completeness_analysis']]
        
        bars = ax1.bar(datasets, completeness_scores, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax1.set_title('Completeness Scores', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Completeness Score', fontsize=10)
        ax1.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, completeness_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Relevance Metrics
        ax2 = axes[0, 1]
        relevance_scores = [analysis['relevance_score'] for analysis in self.df['relevance_analysis']]
        
        bars = ax2.bar(datasets, relevance_scores, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax2.set_title('Relevance Scores', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Relevance Score', fontsize=10)
        ax2.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, relevance_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Consistency Metrics
        ax3 = axes[1, 0]
        consistency_scores = [analysis['overall_consistency'] for analysis in self.df['consistency_analysis']]
        
        bars = ax3.bar(datasets, consistency_scores, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax3.set_title('Consistency Scores', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Consistency Score', fontsize=10)
        ax3.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, consistency_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Diversity Metrics
        ax4 = axes[1, 1]
        diversity_scores = [analysis['diversity_score'] for analysis in self.df['diversity_analysis']]
        
        bars = ax4.bar(datasets, diversity_scores, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax4.set_title('Diversity Scores', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Diversity Score', fontsize=10)
        ax4.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, diversity_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Quality metrics comparison plot saved to {self.output_dir / 'quality_metrics_comparison.png'}")
    
    def plot_detailed_completeness_analysis(self):
        """Plot detailed completeness analysis"""
        print("Generating detailed completeness analysis plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run dataset quality analysis first.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different completeness metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Detailed Completeness Analysis', fontsize=16, fontweight='bold')
        
        datasets = self.df['dataset']
        
        # 1. Feature Completeness
        ax1 = axes[0, 0]
        feature_completeness = [analysis['feature_completeness'] for analysis in self.df['completeness_analysis']]
        
        bars = ax1.bar(datasets, feature_completeness, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax1.set_title('Feature Completeness', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Completeness Score', fontsize=10)
        ax1.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, feature_completeness):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Sample Completeness
        ax2 = axes[0, 1]
        sample_completeness = [analysis['sample_completeness'] for analysis in self.df['completeness_analysis']]
        
        bars = ax2.bar(datasets, sample_completeness, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax2.set_title('Sample Completeness', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Completeness Score', fontsize=10)
        ax2.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, sample_completeness):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Class Completeness
        ax3 = axes[1, 0]
        class_completeness = [analysis['class_completeness'] for analysis in self.df['completeness_analysis']]
        
        bars = ax3.bar(datasets, class_completeness, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax3.set_title('Class Completeness', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Completeness Score', fontsize=10)
        ax3.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, class_completeness):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Missing Features
        ax4 = axes[1, 1]
        missing_features = [analysis['missing_features'] for analysis in self.df['completeness_analysis']]
        
        bars = ax4.bar(datasets, missing_features, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax4.set_title('Missing Features', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Number of Missing Features', fontsize=10)
        
        # Add value labels
        for bar, count in zip(bars, missing_features):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detailed_completeness_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Detailed completeness analysis plot saved to {self.output_dir / 'detailed_completeness_analysis.png'}")
    
    def plot_detailed_relevance_analysis(self):
        """Plot detailed relevance analysis"""
        print("Generating detailed relevance analysis plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run dataset quality analysis first.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different relevance metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Detailed Relevance Analysis', fontsize=16, fontweight='bold')
        
        datasets = self.df['dataset']
        
        # 1. Average Correlation
        ax1 = axes[0, 0]
        avg_correlations = [analysis['average_correlation'] for analysis in self.df['relevance_analysis']]
        
        bars = ax1.bar(datasets, avg_correlations, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax1.set_title('Average Feature Correlation', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Correlation Score', fontsize=10)
        ax1.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, avg_correlations):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Max Correlation
        ax2 = axes[0, 1]
        max_correlations = [analysis['max_correlation'] for analysis in self.df['relevance_analysis']]
        
        bars = ax2.bar(datasets, max_correlations, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax2.set_title('Maximum Feature Correlation', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Correlation Score', fontsize=10)
        ax2.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, max_correlations):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. High Relevance Features
        ax3 = axes[1, 0]
        high_relevance = [int(analysis['high_relevance_features']) for analysis in self.df['relevance_analysis']]
        
        bars = ax3.bar(datasets, high_relevance, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax3.set_title('High Relevance Features', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Features', fontsize=10)
        
        # Add value labels
        for bar, count in zip(bars, high_relevance):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Medium Relevance Features
        ax4 = axes[1, 1]
        medium_relevance = [int(analysis['medium_relevance_features']) for analysis in self.df['relevance_analysis']]
        
        bars = ax4.bar(datasets, medium_relevance, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax4.set_title('Medium Relevance Features', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Number of Features', fontsize=10)
        
        # Add value labels
        for bar, count in zip(bars, medium_relevance):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detailed_relevance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Detailed relevance analysis plot saved to {self.output_dir / 'detailed_relevance_analysis.png'}")
    
    def plot_detailed_consistency_analysis(self):
        """Plot detailed consistency analysis"""
        print("Generating detailed consistency analysis plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run dataset quality analysis first.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different consistency metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Detailed Consistency Analysis', fontsize=16, fontweight='bold')
        
        datasets = self.df['dataset']
        
        # 1. Format Consistency
        ax1 = axes[0, 0]
        format_consistency = [analysis['format_consistency'] for analysis in self.df['consistency_analysis']]
        
        bars = ax1.bar(datasets, format_consistency, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax1.set_title('Format Consistency', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Consistency Score', fontsize=10)
        ax1.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, format_consistency):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Value Consistency
        ax2 = axes[0, 1]
        value_consistency = [analysis['value_consistency'] for analysis in self.df['consistency_analysis']]
        
        bars = ax2.bar(datasets, value_consistency, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax2.set_title('Value Consistency', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Consistency Score', fontsize=10)
        ax2.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, value_consistency):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Range Consistency
        ax3 = axes[1, 0]
        range_consistency = [analysis['range_consistency'] for analysis in self.df['consistency_analysis']]
        
        bars = ax3.bar(datasets, range_consistency, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax3.set_title('Range Consistency', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Consistency Score', fontsize=10)
        ax3.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, range_consistency):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Distribution Consistency
        ax4 = axes[1, 1]
        distribution_consistency = [analysis['distribution_consistency'] for analysis in self.df['consistency_analysis']]
        
        bars = ax4.bar(datasets, distribution_consistency, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax4.set_title('Distribution Consistency', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Consistency Score', fontsize=10)
        ax4.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, distribution_consistency):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detailed_consistency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Detailed consistency analysis plot saved to {self.output_dir / 'detailed_consistency_analysis.png'}")
    
    def plot_detailed_diversity_analysis(self):
        """Plot detailed diversity analysis"""
        print("Generating detailed diversity analysis plot...")
        
        if len(self.df) == 0:
            print("No data to plot. Run dataset quality analysis first.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different diversity metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Detailed Diversity Analysis', fontsize=16, fontweight='bold')
        
        datasets = self.df['dataset']
        
        # 1. Feature Diversity
        ax1 = axes[0, 0]
        feature_diversity = [analysis['feature_diversity'] for analysis in self.df['diversity_analysis']]
        
        bars = ax1.bar(datasets, feature_diversity, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax1.set_title('Feature Diversity', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Diversity Score', fontsize=10)
        ax1.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, feature_diversity):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Class Diversity
        ax2 = axes[0, 1]
        class_diversity = [analysis['class_diversity'] for analysis in self.df['diversity_analysis']]
        
        bars = ax2.bar(datasets, class_diversity, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax2.set_title('Class Diversity', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Diversity Score', fontsize=10)
        ax2.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, class_diversity):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Sample Diversity
        ax3 = axes[1, 0]
        sample_diversity = [analysis['sample_diversity'] for analysis in self.df['diversity_analysis']]
        
        bars = ax3.bar(datasets, sample_diversity, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax3.set_title('Sample Diversity', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Diversity Score', fontsize=10)
        ax3.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, sample_diversity):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Pattern Diversity
        ax4 = axes[1, 1]
        pattern_diversity = [analysis['pattern_diversity'] for analysis in self.df['diversity_analysis']]
        
        bars = ax4.bar(datasets, pattern_diversity, 
                       color=[self.dataset_colors.get(d.lower(), '#666666') for d in datasets])
        ax4.set_title('Pattern Diversity', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Diversity Score', fontsize=10)
        ax4.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, pattern_diversity):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detailed_diversity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Detailed diversity analysis plot saved to {self.output_dir / 'detailed_diversity_analysis.png'}")
    
    def plot_quality_heatmap(self):
        """Plot quality metrics heatmap"""
        print("Generating quality metrics heatmap...")
        
        if len(self.df) == 0:
            print("No data to plot. Run dataset quality analysis first.")
            return
        
        # Create heatmap data
        heatmap_data = []
        datasets = self.df['dataset']
        
        for _, row in self.df.iterrows():
            dataset = row['dataset']
            completeness = row['completeness_analysis']['overall_completeness']
            relevance = row['relevance_analysis']['relevance_score']
            consistency = row['consistency_analysis']['overall_consistency']
            diversity = row['diversity_analysis']['diversity_score']
            
            heatmap_data.append([completeness, relevance, consistency, diversity])
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                index=datasets,
                                columns=['Completeness', 'Relevance', 'Consistency', 'Diversity'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Quality Score'})
        plt.title('Dataset Quality Metrics Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Quality Metrics', fontsize=12)
        plt.ylabel('Dataset', fontsize=12)
        
        plt.savefig(self.output_dir / 'quality_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Quality heatmap saved to {self.output_dir / 'quality_heatmap.png'}")
    
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
        fig.suptitle('Dataset Quality Summary Statistics', fontsize=16, fontweight='bold')
        
        # 1. Overall quality scores
        ax1 = axes[0, 0]
        datasets = summary.get('datasets_analyzed', [])
        avg_quality = summary.get('average_quality_score', 0)
        quality_std = summary.get('quality_score_std', 0)
        
        ax1.bar(['Average'], [avg_quality], yerr=[quality_std], capsize=5, color='#4ECDC4')
        ax1.set_title('Average Quality Score', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Quality Score', fontsize=10)
        ax1.set_ylim(0, 1.1)
        
        # Add value label
        ax1.text(0, avg_quality + quality_std + 0.01,
                f'{avg_quality:.3f}±{quality_std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Completeness statistics
        ax2 = axes[0, 1]
        completeness_stats = summary.get('completeness_stats', {})
        avg_completeness = completeness_stats.get('average', 0)
        completeness_std = completeness_stats.get('std', 0)
        
        ax2.bar(['Average'], [avg_completeness], yerr=[completeness_std], capsize=5, color='#45B7D1')
        ax2.set_title('Average Completeness Score', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Completeness Score', fontsize=10)
        ax2.set_ylim(0, 1.1)
        
        # Add value label
        ax2.text(0, avg_completeness + completeness_std + 0.01,
                f'{avg_completeness:.3f}±{completeness_std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Relevance statistics
        ax3 = axes[1, 0]
        relevance_stats = summary.get('relevance_stats', {})
        avg_relevance = relevance_stats.get('average', 0)
        relevance_std = relevance_stats.get('std', 0)
        
        ax3.bar(['Average'], [avg_relevance], yerr=[relevance_std], capsize=5, color='#96CEB4')
        ax3.set_title('Average Relevance Score', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Relevance Score', fontsize=10)
        ax3.set_ylim(0, 1.1)
        
        # Add value label
        ax3.text(0, avg_relevance + relevance_std + 0.01,
                f'{avg_relevance:.3f}±{relevance_std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Consistency statistics
        ax4 = axes[1, 1]
        consistency_stats = summary.get('consistency_stats', {})
        avg_consistency = consistency_stats.get('average', 0)
        consistency_std = consistency_stats.get('std', 0)
        
        ax4.bar(['Average'], [avg_consistency], yerr=[consistency_std], capsize=5, color='#FFEAA7')
        ax4.set_title('Average Consistency Score', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Consistency Score', fontsize=10)
        ax4.set_ylim(0, 1.1)
        
        # Add value label
        ax4.text(0, avg_consistency + consistency_std + 0.01,
                f'{avg_consistency:.3f}±{consistency_std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary statistics plot saved to {self.output_dir / 'summary_statistics.png'}")
    
    def generate_all_plots(self):
        """Generate all dataset quality plots"""
        print("Generating all dataset quality visualization plots...")
        
        # Generate all plots
        self.plot_quality_overview()
        self.plot_quality_metrics_comparison()
        self.plot_detailed_completeness_analysis()
        self.plot_detailed_relevance_analysis()
        self.plot_detailed_consistency_analysis()
        self.plot_detailed_diversity_analysis()
        self.plot_quality_heatmap()
        self.plot_summary_statistics()
        
        print(f"\nAll plots generated successfully in {self.output_dir}")
        print("Generated plots:")
        plot_files = [
            'quality_overview.png',
            'quality_metrics_comparison.png',
            'detailed_completeness_analysis.png',
            'detailed_relevance_analysis.png',
            'detailed_consistency_analysis.png',
            'detailed_diversity_analysis.png',
            'quality_heatmap.png',
            'summary_statistics.png'
        ]
        
        for plot_file in plot_files:
            if (self.output_dir / plot_file).exists():
                print(f"  ✅ {plot_file}")
            else:
                print(f"  ❌ {plot_file} (no data available)")

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Generate Dataset Quality Visualization Plots')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Input dataset quality results JSON file')
    parser.add_argument('--output_dir', type=str, default='./dataset_quality_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create plotter
    plotter = DatasetQualityPlotter(args.input_file, args.output_dir)
    
    # Generate all plots
    plotter.generate_all_plots()
    
    print(f"\nDataset quality plots generation completed!")
    print(f"Output directory: {args.output_dir}")

if __name__ == '__main__':
    main() 