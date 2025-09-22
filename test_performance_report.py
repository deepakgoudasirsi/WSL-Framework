#!/usr/bin/env python3
"""
Test script to demonstrate the performance report generator
"""

# Import warning suppression utility first
from src.utils.warning_suppression import apply_warning_suppression
apply_warning_suppression()

import json
from pathlib import Path
from src.experiments.generate_performance_report import PerformanceReportGenerator

def create_sample_results():
    """Create sample experiment results for testing"""
    print("Creating sample experiment results...")
    
    # Create sample experiment directories
    sample_experiments = [
        {
            'dataset': 'mnist',
            'model_type': 'mlp',
            'strategy': 'consistency',
            'accuracy': 98.17,
            'training_time': 35,
            'memory_usage': 2.0
        },
        {
            'dataset': 'mnist',
            'model_type': 'mlp',
            'strategy': 'pseudo_label',
            'accuracy': 98.26,
            'training_time': 42,
            'memory_usage': 2.3
        },
        {
            'dataset': 'cifar10',
            'model_type': 'simple_cnn',
            'strategy': 'traditional',
            'accuracy': 82.1,
            'training_time': 90,
            'memory_usage': 2.3
        },
        {
            'dataset': 'cifar10',
            'model_type': 'resnet',
            'strategy': 'combined',
            'accuracy': 89.3,
            'training_time': 450,
            'memory_usage': 4.2
        }
    ]
    
    # Create experiment directories and files
    for exp in sample_experiments:
        # Create semi-supervised experiment directory
        if exp['strategy'] in ['consistency', 'pseudo_label', 'co_training']:
            exp_dir = Path(f"experiments/semi_supervised/{exp['dataset']}_{exp['model_type']}_labeled0.1_sample")
        else:
            exp_dir = Path(f"experiments/noise_robustness/noise_robustness_{exp['dataset']}_{exp['model_type']}_sample")
        
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create config file
        config = {
            'dataset': exp['dataset'],
            'model_type': exp['model_type'],
            'strategy': exp['strategy'],
            'epochs': 50,
            'batch_size': 128,
            'learning_rate': 0.001
        }
        
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create results file
        results = {
            'best_val_acc': exp['accuracy'],
            'final_test_acc': exp['accuracy'],
            'training_time': exp['training_time'],
            'memory_usage': exp['memory_usage'],
            'metrics': {
                'val_acc': [exp['accuracy'] - 2, exp['accuracy'] - 1, exp['accuracy']],
                'train_acc': [exp['accuracy'] - 1, exp['accuracy'] - 0.5, exp['accuracy']]
            }
        }
        
        with open(exp_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"Created {len(sample_experiments)} sample experiment results")

def test_performance_report():
    """Test the performance report generator with sample data"""
    print("Testing performance report generator...")
    
    # Create sample results
    create_sample_results()
    
    # Generate performance report
    generator = PerformanceReportGenerator('test_performance_report.json')
    
    # Collect results
    generator.collect_experiment_results(
        datasets=['mnist', 'cifar10'],
        model_types=['mlp', 'simple_cnn', 'resnet'],
        strategies=['traditional', 'consistency', 'pseudo_label', 'combined']
    )
    
    # Save report
    generator.save_report()
    
    print("Test completed successfully!")

if __name__ == '__main__':
    test_performance_report() 