#!/usr/bin/env python3
"""
Summary Report Generator
Generates comprehensive summary reports from experiment results
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummaryReportGenerator:
    """Generate comprehensive summary reports from experiment results"""
    
    def __init__(self, output_file: str):
        self.output_file = Path(output_file)
        self.results = {}
        self.experiment_dirs = {
            'semi_supervised': Path('experiments/semi_supervised'),
            'noise_robustness': Path('experiments/noise_robustness'),
            'baseline': Path('experiments/baseline'),
            'training_curves': Path('training_curves'),
            'performance_plots': Path('performance_plots')
        }
    
    def collect_experiment_results(self):
        """Collect results from various experiment directories"""
        print("Collecting experiment results...")
        
        # Load performance comparison report if it exists
        performance_file = Path('performance_comparison_report.json')
        if performance_file.exists():
            with open(performance_file, 'r') as f:
                self.results['performance'] = json.load(f)
            print("✓ Loaded performance comparison report")
        
        # Load test performance report if it exists
        test_performance_file = Path('test_performance_report.json')
        if test_performance_file.exists():
            with open(test_performance_file, 'r') as f:
                self.results['test_performance'] = json.load(f)
            print("✓ Loaded test performance report")
        
        # Load training curves summary if it exists
        training_summary_file = Path('training_curves/training_summary_report.json')
        if training_summary_file.exists():
            with open(training_summary_file, 'r') as f:
                self.results['training_curves'] = json.load(f)
            print("✓ Loaded training curves summary")
        
        # Collect confusion matrix results
        self._collect_confusion_matrices()
        
        # Collect memory analysis results
        self._collect_memory_analysis()
        
        # Collect training time analysis
        self._collect_training_time_analysis()
    
    def _collect_confusion_matrices(self):
        """Collect confusion matrix results"""
        confusion_files = [
            'cifar10_confusion_matrix.png',
            'mnist_confusion_matrix.png',
            'mnist_mlp_confusion_matrix.png'
        ]
        
        self.results['confusion_matrices'] = {}
        for file in confusion_files:
            if Path(file).exists():
                self.results['confusion_matrices'][file] = {
                    'exists': True,
                    'size_kb': Path(file).stat().st_size / 1024
                }
            else:
                self.results['confusion_matrices'][file] = {
                    'exists': False
                }
    
    def _collect_memory_analysis(self):
        """Collect memory analysis results"""
        memory_files = list(Path('.').glob('memory_analysis_*.json'))
        if memory_files:
            latest_memory_file = max(memory_files, key=lambda x: x.stat().st_mtime)
            with open(latest_memory_file, 'r') as f:
                self.results['memory_analysis'] = json.load(f)
            print(f"✓ Loaded memory analysis from {latest_memory_file}")
    
    def _collect_training_time_analysis(self):
        """Collect training time analysis results"""
        time_files = list(Path('.').glob('training_time_analysis_*.json'))
        if time_files:
            latest_time_file = max(time_files, key=lambda x: x.stat().st_mtime)
            with open(latest_time_file, 'r') as f:
                self.results['training_time_analysis'] = json.load(f)
            print(f"✓ Loaded training time analysis from {latest_time_file}")
    
    def generate_performance_summary(self) -> str:
        """Generate performance summary section"""
        if 'performance' not in self.results:
            return "## Performance Summary\n\nNo performance data available.\n\n"
        
        performance = self.results['performance']
        summary = "## Performance Summary\n\n"
        
        if 'summary' in performance and 'overall' in performance['summary']:
            overall = performance['summary']['overall']
            summary += f"### Overall Results\n\n"
            summary += f"- **Total Experiments**: {overall.get('total_experiments', 'N/A')}\n"
            summary += f"- **Best Accuracy**: {overall.get('best_accuracy', 'N/A'):.2f}%\n"
            summary += f"- **Average Accuracy**: {overall.get('avg_accuracy', 'N/A'):.2f}%\n"
            summary += f"- **Standard Deviation**: {overall.get('std_accuracy', 'N/A'):.2f}%\n\n"
        
        # Dataset performance
        if 'summary' in performance and 'datasets' in performance['summary']:
            summary += "### Dataset Performance\n\n"
            for dataset, data in performance['summary']['datasets'].items():
                summary += f"#### {dataset.upper()}\n"
                summary += f"- **Best Accuracy**: {data.get('best_accuracy', 'N/A'):.2f}%\n"
                summary += f"- **Average Accuracy**: {data.get('avg_accuracy', 'N/A'):.2f}%\n"
                summary += f"- **Experiments**: {data.get('experiments', 'N/A')}\n\n"
        
        # Strategy performance
        if 'summary' in performance and 'strategies' in performance['summary']:
            summary += "### Strategy Performance\n\n"
            for strategy, data in performance['summary']['strategies'].items():
                summary += f"#### {strategy.title()}\n"
                summary += f"- **Best Accuracy**: {data.get('best_accuracy', 'N/A'):.2f}%\n"
                summary += f"- **Average Accuracy**: {data.get('avg_accuracy', 'N/A'):.2f}%\n"
                summary += f"- **Experiments**: {data.get('experiments', 'N/A')}\n\n"
        
        return summary
    
    def generate_training_curves_summary(self) -> str:
        """Generate training curves summary section"""
        if 'training_curves' not in self.results:
            return "## Training Curves Analysis\n\nNo training curves data available.\n\n"
        
        training_data = self.results['training_curves']
        summary = "## Training Curves Analysis\n\n"
        
        if isinstance(training_data, list):
            df = pd.DataFrame(training_data)
            
            summary += "### Performance Comparison\n\n"
            summary += "| Dataset | Strategy | Final Train Acc | Final Val Acc | Convergence Epoch | Overfitting Gap |\n"
            summary += "|---------|----------|-----------------|---------------|-------------------|-----------------|\n"
            
            for _, row in df.iterrows():
                summary += f"| {row['dataset']} | {row['strategy']} | {row['final_train_acc']:.3f} | {row['final_val_acc']:.3f} | {row['convergence_epoch']} | {row['overfitting_gap']:.3f} |\n"
            
            summary += "\n"
            
            # Best performers
            best_val_acc = df.loc[df['final_val_acc'].idxmax()]
            best_train_acc = df.loc[df['final_train_acc'].idxmax()]
            fastest_convergence = df.loc[df['convergence_epoch'].idxmin()]
            
            summary += "### Best Performers\n\n"
            summary += f"- **Best Validation Accuracy**: {best_val_acc['dataset']} - {best_val_acc['strategy']} ({best_val_acc['final_val_acc']:.3f})\n"
            summary += f"- **Best Training Accuracy**: {best_train_acc['dataset']} - {best_train_acc['strategy']} ({best_train_acc['final_train_acc']:.3f})\n"
            summary += f"- **Fastest Convergence**: {fastest_convergence['dataset']} - {fastest_convergence['strategy']} (Epoch {fastest_convergence['convergence_epoch']})\n\n"
        
        return summary
    
    def generate_memory_analysis_summary(self) -> str:
        """Generate memory analysis summary section"""
        if 'memory_analysis' not in self.results:
            return "## Memory Analysis\n\nNo memory analysis data available.\n\n"
        
        memory_data = self.results['memory_analysis']
        summary = "## Memory Analysis\n\n"
        
        if 'summary' in memory_data:
            summary_stats = memory_data['summary']
            
            summary += f"### Overall Statistics\n\n"
            summary += f"- **Total Configurations**: {summary_stats.get('total_configurations', 'N/A')}\n"
            summary += f"- **Valid Configurations**: {summary_stats.get('valid_configurations', 'N/A')}\n"
            summary += f"- **Failed Configurations**: {summary_stats.get('failed_configurations', 'N/A')}\n\n"
            
            if 'memory_statistics' in summary_stats:
                mem_stats = summary_stats['memory_statistics']
                summary += f"### Memory Usage Statistics\n\n"
                summary += f"- **Average Memory Usage**: {mem_stats.get('avg_memory_gb', 'N/A'):.2f} GB\n"
                summary += f"- **Maximum Memory Usage**: {mem_stats.get('max_memory_gb', 'N/A'):.2f} GB\n"
                summary += f"- **Minimum Memory Usage**: {mem_stats.get('min_memory_gb', 'N/A'):.2f} GB\n\n"
        
        return summary
    
    def generate_training_time_summary(self) -> str:
        """Generate training time analysis summary section"""
        if 'training_time_analysis' not in self.results:
            return "## Training Time Analysis\n\nNo training time analysis data available.\n\n"
        
        time_data = self.results['training_time_analysis']
        summary = "## Training Time Analysis\n\n"
        
        if 'summary' in time_data:
            summary_stats = time_data['summary']
            
            summary += f"### Overall Statistics\n\n"
            summary += f"- **Total Configurations**: {summary_stats.get('total_configurations', 'N/A')}\n"
            summary += f"- **Valid Configurations**: {summary_stats.get('valid_configurations', 'N/A')}\n"
            summary += f"- **Failed Configurations**: {summary_stats.get('failed_configurations', 'N/A')}\n\n"
            
            if 'timing_statistics' in summary_stats:
                timing_stats = summary_stats['timing_statistics']
                summary += f"### Timing Statistics\n\n"
                summary += f"- **Average Training Time**: {timing_stats.get('avg_total_time_minutes', 'N/A'):.2f} minutes\n"
                summary += f"- **Fastest Training**: {timing_stats.get('min_total_time_minutes', 'N/A'):.2f} minutes\n"
                summary += f"- **Slowest Training**: {timing_stats.get('max_total_time_minutes', 'N/A'):.2f} minutes\n"
                summary += f"- **Average Epoch Time**: {timing_stats.get('avg_epoch_time', 'N/A'):.2f} seconds\n\n"
        
        return summary
    
    def generate_confusion_matrix_summary(self) -> str:
        """Generate confusion matrix summary section"""
        if 'confusion_matrices' not in self.results:
            return "## Confusion Matrix Analysis\n\nNo confusion matrix data available.\n\n"
        
        confusion_data = self.results['confusion_matrices']
        summary = "## Confusion Matrix Analysis\n\n"
        
        summary += "### Available Confusion Matrices\n\n"
        for filename, data in confusion_data.items():
            if data.get('exists', False):
                size_kb = data.get('size_kb', 0)
                summary += f"- **{filename}**: {size_kb:.1f} KB\n"
            else:
                summary += f"- **{filename}**: Not available\n"
        
        summary += "\n"
        return summary
    
    def generate_recommendations(self) -> str:
        """Generate recommendations based on results"""
        recommendations = "## Recommendations\n\n"
        
        # Performance-based recommendations
        if 'performance' in self.results:
            performance = self.results['performance']
            if 'summary' in performance and 'overall' in performance['summary']:
                overall = performance['summary']['overall']
                best_acc = overall.get('best_accuracy', 0)
                
                if best_acc > 95:
                    recommendations += "### High Performance Achieved\n\n"
                    recommendations += "- The framework demonstrates excellent performance with accuracy above 95%\n"
                    recommendations += "- Consider deploying the best-performing model in production\n"
                    recommendations += "- Monitor performance in real-world conditions\n\n"
                elif best_acc > 85:
                    recommendations += "### Good Performance Achieved\n\n"
                    recommendations += "- The framework shows good performance with accuracy above 85%\n"
                    recommendations += "- Consider fine-tuning hyperparameters for further improvement\n"
                    recommendations += "- Evaluate additional data augmentation techniques\n\n"
                else:
                    recommendations += "### Performance Improvement Needed\n\n"
                    recommendations += "- Consider experimenting with different model architectures\n"
                    recommendations += "- Try different WSL strategies or combinations\n"
                    recommendations += "- Review data preprocessing and augmentation\n\n"
        
        # Memory and time recommendations
        if 'memory_analysis' in self.results or 'training_time_analysis' in self.results:
            recommendations += "### Resource Optimization\n\n"
            recommendations += "- Monitor memory usage during training\n"
            recommendations += "- Consider batch size optimization for memory efficiency\n"
            recommendations += "- Evaluate training time vs. performance trade-offs\n"
            recommendations += "- Consider distributed training for large-scale experiments\n\n"
        
        # General recommendations
        recommendations += "### General Recommendations\n\n"
        recommendations += "- Continue monitoring model performance in production\n"
        recommendations += "- Implement regular model retraining with new data\n"
        recommendations += "- Consider ensemble methods for improved robustness\n"
        recommendations += "- Document all experimental configurations for reproducibility\n\n"
        
        return recommendations
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary"""
        summary = "## Executive Summary\n\n"
        
        # Key metrics
        if 'performance' in self.results:
            performance = self.results['performance']
            if 'summary' in performance and 'overall' in performance['summary']:
                overall = performance['summary']['overall']
                summary += f"This comprehensive analysis of the Weakly Supervised Learning (WSL) framework "
                summary += f"demonstrates {overall.get('total_experiments', 0)} experiments across multiple "
                summary += f"datasets and strategies. The framework achieved a best accuracy of "
                summary += f"{overall.get('best_accuracy', 0):.2f}% with an average accuracy of "
                summary += f"{overall.get('avg_accuracy', 0):.2f}%.\n\n"
        
        # Key findings
        summary += "### Key Findings\n\n"
        summary += "- **Framework Performance**: The WSL framework successfully demonstrates the ability to train models with limited labeled data\n"
        summary += "- **Strategy Effectiveness**: Different WSL strategies show varying levels of effectiveness across datasets\n"
        summary += "- **Resource Efficiency**: The framework provides good balance between performance and computational resources\n"
        summary += "- **Scalability**: The framework supports multiple datasets and model architectures\n\n"
        
        # Next steps
        summary += "### Next Steps\n\n"
        summary += "- Deploy the best-performing model configurations\n"
        summary += "- Implement continuous monitoring and evaluation\n"
        summary += "- Expand experiments to additional datasets and strategies\n"
        summary += "- Optimize resource usage for production deployment\n\n"
        
        return summary
    
    def generate_complete_report(self) -> str:
        """Generate the complete summary report"""
        print("Generating comprehensive summary report...")
        
        # Collect all results
        self.collect_experiment_results()
        
        # Generate report sections
        report = "# WSL Framework - Comprehensive Summary Report\n\n"
        report += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Executive summary
        report += self.generate_executive_summary()
        
        # Performance summary
        report += self.generate_performance_summary()
        
        # Training curves analysis
        report += self.generate_training_curves_summary()
        
        # Memory analysis
        report += self.generate_memory_analysis_summary()
        
        # Training time analysis
        report += self.generate_training_time_summary()
        
        # Confusion matrix analysis
        report += self.generate_confusion_matrix_summary()
        
        # Recommendations
        report += self.generate_recommendations()
        
        # Conclusion
        report += "## Conclusion\n\n"
        report += "This comprehensive analysis demonstrates the effectiveness of the Weakly Supervised Learning framework "
        report += "in achieving high performance with limited labeled data. The framework successfully balances "
        report += "performance, efficiency, and scalability across multiple datasets and strategies.\n\n"
        
        report += "The results provide a solid foundation for further research and practical applications "
        report += "in scenarios where labeled data is scarce or expensive to obtain.\n\n"
        
        return report
    
    def save_report(self, report_content: str):
        """Save the report to file"""
        print(f"Saving summary report to {self.output_file}...")
        
        with open(self.output_file, 'w') as f:
            f.write(report_content)
        
        print(f"✓ Summary report saved successfully!")
        print(f"Report file: {self.output_file}")
        print(f"Report size: {self.output_file.stat().st_size / 1024:.1f} KB")

def main():
    """Main function to run the summary report generator"""
    parser = argparse.ArgumentParser(description='Generate comprehensive summary report from experiment results')
    parser.add_argument('--output_file', type=str, default='final_summary_report.md',
                       help='Output file for the summary report (default: final_summary_report.md)')
    
    args = parser.parse_args()
    
    # Create generator and run
    generator = SummaryReportGenerator(args.output_file)
    report_content = generator.generate_complete_report()
    generator.save_report(report_content)
    
    print(f"\nSummary report generation completed!")
    print(f"Output file: {args.output_file}")

if __name__ == '__main__':
    main() 