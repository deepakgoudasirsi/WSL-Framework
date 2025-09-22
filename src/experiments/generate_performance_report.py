#!/usr/bin/env python3
"""
Performance Report Generator
Generates comprehensive performance comparison reports from experiment results
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

class PerformanceReportGenerator:
    """Generate comprehensive performance reports from experiment results"""
    
    def __init__(self, output_file: str):
        self.output_file = Path(output_file)
        self.results = {}
        self.experiment_dirs = {
            'semi_supervised': Path('experiments/semi_supervised'),
            'noise_robustness': Path('experiments/noise_robustness'),
            'baseline': Path('experiments/baseline')
        }
    
    def collect_experiment_results(self, datasets: List[str], model_types: List[str], strategies: List[str]):
        """Collect results from experiment directories"""
        print("Collecting experiment results...")
        
        for dataset in datasets:
            self.results[dataset] = {}
            
            for model_type in model_types:
                self.results[dataset][model_type] = {}
                
                for strategy in strategies:
                    # Look for experiment results
                    experiment_results = self._find_experiment_results(dataset, model_type, strategy)
                    
                    if experiment_results:
                        self.results[dataset][model_type][strategy] = experiment_results
                        print(f"Found results for {dataset}-{model_type}-{strategy}")
                    else:
                        print(f"No results found for {dataset}-{model_type}-{strategy}")
    
    def _find_experiment_results(self, dataset: str, model_type: str, strategy: str) -> Dict[str, Any]:
        """Find experiment results for a specific configuration"""
        # Look in semi-supervised experiments
        if strategy in ['consistency', 'pseudo_label', 'co_training', 'mixmatch']:
            experiment_dir = self.experiment_dirs['semi_supervised']
            pattern = f"{dataset}_{model_type}_labeled0.1_*"
            
            for exp_dir in experiment_dir.glob(pattern):
                config_file = exp_dir / 'config.json'
                results_file = exp_dir / 'results.json'
                
                if config_file.exists() and results_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    if config.get('strategy') == strategy:
                        with open(results_file, 'r') as f:
                            results = json.load(f)
                        
                        return {
                            'config': config,
                            'results': results,
                            'experiment_dir': str(exp_dir)
                        }
        
        # Look in noise robustness experiments
        elif strategy == 'traditional':
            experiment_dir = self.experiment_dirs['noise_robustness']
            pattern = f"noise_robustness_{dataset}_{model_type}_*"
            
            for exp_dir in experiment_dir.glob(pattern):
                config_file = exp_dir / 'config.json'
                results_file = exp_dir / 'results.json'
                
                if config_file.exists() and results_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    return {
                        'config': config,
                        'results': results,
                        'experiment_dir': str(exp_dir)
                    }
        
        return None
    
    def generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary statistics"""
        print("Generating performance summary...")
        
        summary = {
            'datasets': {},
            'model_types': {},
            'strategies': {},
            'overall': {}
        }
        
        # Collect all accuracies
        all_accuracies = []
        
        for dataset in self.results:
            summary['datasets'][dataset] = {
                'best_accuracy': 0,
                'avg_accuracy': 0,
                'experiments': 0
            }
            
            dataset_accuracies = []
            
            for model_type in self.results[dataset]:
                if model_type not in summary['model_types']:
                    summary['model_types'][model_type] = {
                        'best_accuracy': 0,
                        'avg_accuracy': 0,
                        'experiments': 0
                    }
                
                model_accuracies = []
                
                for strategy in self.results[dataset][model_type]:
                    if strategy not in summary['strategies']:
                        summary['strategies'][strategy] = {
                            'best_accuracy': 0,
                            'avg_accuracy': 0,
                            'experiments': 0
                        }
                    
                    experiment_data = self.results[dataset][model_type][strategy]
                    
                    # Extract accuracy from results
                    if 'results' in experiment_data:
                        results = experiment_data['results']
                        
                        # Try different possible locations for accuracy
                        accuracy = None
                        if 'final_test_acc' in results:
                            accuracy = results['final_test_acc']
                        elif 'best_val_acc' in results:
                            accuracy = results['best_val_acc']
                        elif 'metrics' in results and 'val_acc' in results['metrics']:
                            accuracy = max(results['metrics']['val_acc'])
                        
                        if accuracy is not None:
                            all_accuracies.append(accuracy)
                            dataset_accuracies.append(accuracy)
                            model_accuracies.append(accuracy)
                            
                            # Update strategy summary
                            summary['strategies'][strategy]['experiments'] += 1
                            summary['strategies'][strategy]['best_accuracy'] = max(
                                summary['strategies'][strategy]['best_accuracy'], accuracy
                            )
                
                # Update model type summary
                if model_accuracies:
                    summary['model_types'][model_type]['experiments'] += len(model_accuracies)
                    summary['model_types'][model_type]['best_accuracy'] = max(
                        summary['model_types'][model_type]['best_accuracy'], max(model_accuracies)
                    )
                    summary['model_types'][model_type]['avg_accuracy'] = np.mean(model_accuracies)
            
            # Update dataset summary
            if dataset_accuracies:
                summary['datasets'][dataset]['experiments'] = len(dataset_accuracies)
                summary['datasets'][dataset]['best_accuracy'] = max(dataset_accuracies)
                summary['datasets'][dataset]['avg_accuracy'] = np.mean(dataset_accuracies)
        
        # Update overall summary
        if all_accuracies:
            summary['overall'] = {
                'total_experiments': len(all_accuracies),
                'best_accuracy': max(all_accuracies),
                'avg_accuracy': np.mean(all_accuracies),
                'std_accuracy': np.std(all_accuracies)
            }
        
        return summary
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate comparison table for all experiments"""
        print("Generating comparison table...")
        
        table_data = []
        
        for dataset in self.results:
            for model_type in self.results[dataset]:
                for strategy in self.results[dataset][model_type]:
                    experiment_data = self.results[dataset][model_type][strategy]
                    
                    # Extract metrics
                    accuracy = None
                    training_time = None
                    memory_usage = None
                    
                    if 'results' in experiment_data:
                        results = experiment_data['results']
                        
                        # Extract accuracy
                        if 'final_test_acc' in results:
                            accuracy = results['final_test_acc']
                        elif 'best_val_acc' in results:
                            accuracy = results['best_val_acc']
                        elif 'metrics' in results and 'val_acc' in results['metrics']:
                            accuracy = max(results['metrics']['val_acc'])
                        
                        # Extract training time (if available)
                        if 'training_time' in results:
                            training_time = results['training_time']
                        
                        # Extract memory usage (if available)
                        if 'memory_usage' in results:
                            memory_usage = results['memory_usage']
                    
                    table_data.append({
                        'Dataset': dataset,
                        'Model_Type': model_type,
                        'Strategy': strategy,
                        'Accuracy': accuracy,
                        'Training_Time_Min': training_time,
                        'Memory_Usage_GB': memory_usage,
                        'Experiment_Dir': experiment_data.get('experiment_dir', '')
                    })
        
        return pd.DataFrame(table_data)
    
    def generate_ranking_analysis(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate ranking analysis for different categories"""
        print("Generating ranking analysis...")
        
        rankings = {
            'by_accuracy': [],
            'by_dataset': {},
            'by_model_type': {},
            'by_strategy': {}
        }
        
        # Collect all experiments with accuracy
        experiments_with_accuracy = []
        
        for dataset in self.results:
            for model_type in self.results[dataset]:
                for strategy in self.results[dataset][model_type]:
                    experiment_data = self.results[dataset][model_type][strategy]
                    
                    if 'results' in experiment_data:
                        results = experiment_data['results']
                        
                        # Extract accuracy
                        accuracy = None
                        if 'final_test_acc' in results:
                            accuracy = results['final_test_acc']
                        elif 'best_val_acc' in results:
                            accuracy = results['best_val_acc']
                        elif 'metrics' in results and 'val_acc' in results['metrics']:
                            accuracy = max(results['metrics']['val_acc'])
                        
                        if accuracy is not None:
                            experiment_info = {
                                'dataset': dataset,
                                'model_type': model_type,
                                'strategy': strategy,
                                'accuracy': accuracy,
                                'experiment_dir': experiment_data.get('experiment_dir', '')
                            }
                            experiments_with_accuracy.append(experiment_info)
        
        # Sort by accuracy
        experiments_with_accuracy.sort(key=lambda x: x['accuracy'], reverse=True)
        rankings['by_accuracy'] = experiments_with_accuracy
        
        # Rankings by dataset
        for dataset in self.results:
            dataset_experiments = [exp for exp in experiments_with_accuracy if exp['dataset'] == dataset]
            dataset_experiments.sort(key=lambda x: x['accuracy'], reverse=True)
            rankings['by_dataset'][dataset] = dataset_experiments
        
        # Rankings by model type
        model_types = set(exp['model_type'] for exp in experiments_with_accuracy)
        for model_type in model_types:
            model_experiments = [exp for exp in experiments_with_accuracy if exp['model_type'] == model_type]
            model_experiments.sort(key=lambda x: x['accuracy'], reverse=True)
            rankings['by_model_type'][model_type] = model_experiments
        
        # Rankings by strategy
        strategies = set(exp['strategy'] for exp in experiments_with_accuracy)
        for strategy in strategies:
            strategy_experiments = [exp for exp in experiments_with_accuracy if exp['strategy'] == strategy]
            strategy_experiments.sort(key=lambda x: x['accuracy'], reverse=True)
            rankings['by_strategy'][strategy] = strategy_experiments
        
        return rankings
    
    def save_report(self):
        """Save the complete performance report"""
        print(f"Saving performance report to {self.output_file}...")
        
        # Generate all components
        summary = self.generate_performance_summary()
        comparison_table = self.generate_comparison_table()
        rankings = self.generate_ranking_analysis()
        
        # Create complete report
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_experiments': len(comparison_table),
                'datasets': list(self.results.keys()),
                'model_types': list(set(comparison_table['Model_Type'].dropna())) if len(comparison_table) > 0 else [],
                'strategies': list(set(comparison_table['Strategy'].dropna())) if len(comparison_table) > 0 else []
            },
            'summary': summary,
            'comparison_table': comparison_table.to_dict('records'),
            'rankings': rankings,
            'raw_results': self.results
        }
        
        # Save to file
        with open(self.output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Performance report saved successfully!")
        print(f"Total experiments analyzed: {len(comparison_table)}")
        
        if summary['overall']:
            print(f"Best overall accuracy: {summary['overall'].get('best_accuracy', 'N/A')}%")
        else:
            print("No experiment results found. Run some experiments first!")
        
        # Print top 5 results
        if rankings['by_accuracy']:
            print("\nTop 5 Results:")
            for i, exp in enumerate(rankings['by_accuracy'][:5]):
                print(f"{i+1}. {exp['dataset']}-{exp['model_type']}-{exp['strategy']}: {exp['accuracy']:.2f}%")
        else:
            print("\nNo results to display. Run experiments to generate performance data.")

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Generate Performance Report')
    parser.add_argument('--datasets', nargs='+', default=['cifar10', 'mnist'],
                       help='Datasets to analyze')
    parser.add_argument('--model_types', nargs='+', 
                       default=['simple_cnn', 'robust_cnn', 'resnet', 'robust_resnet', 'mlp', 'robust_mlp'],
                       help='Model types to analyze')
    parser.add_argument('--strategies', nargs='+',
                       default=['traditional', 'consistency', 'pseudo_label', 'co_training', 'combined'],
                       help='Strategies to analyze')
    parser.add_argument('--output_file', type=str, default='performance_comparison_report.json',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Create report generator
    generator = PerformanceReportGenerator(args.output_file)
    
    # Collect results
    generator.collect_experiment_results(args.datasets, args.model_types, args.strategies)
    
    # Generate and save report
    generator.save_report()

if __name__ == '__main__':
    main() 