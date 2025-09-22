import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .wsl_strategies import (
    WSLStrategy, DataProgrammingStrategy, NoiseRobustStrategy, 
    AdaptiveLearning, ModelSelector, UnifiedFramework
)
from .performance_optimizer import (
    PerformanceOptimizer, OptimizationConfig, 
    AdaptiveBatchSizeOptimizer, LearningRateOptimizer
)

@dataclass
class UnifiedFrameworkConfig:
    """Configuration for enhanced unified framework"""
    # Strategy configuration
    strategies: List[str] = None  # ['data_programming', 'noise_robust']
    strategy_weights: List[float] = None
    
    # Model configuration
    model_type: str = 'robust_cnn'
    num_classes: int = 10
    
    # Training configuration
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Optimization configuration
    use_mixed_precision: bool = True
    use_gradient_accumulation: bool = True
    accumulation_steps: int = 4
    use_early_stopping: bool = True
    patience: int = 10
    
    # Evaluation configuration
    evaluation_metrics: List[str] = None  # ['accuracy', 'f1', 'precision', 'recall']
    cross_validation_folds: int = 5
    
    # Output configuration
    save_dir: str = 'experiments'
    save_model: bool = True
    save_results: bool = True
    create_visualizations: bool = True
    
    def __post_init__(self):
        if self.strategies is None:
            self.strategies = ['data_programming', 'noise_robust']
        if self.strategy_weights is None:
            self.strategy_weights = [1.0] * len(self.strategies)
        if self.evaluation_metrics is None:
            self.evaluation_metrics = ['accuracy', 'f1', 'precision', 'recall']

class EnhancedUnifiedFramework:
    """
    Enhanced Unified Framework for Weakly Supervised Learning
    
    This framework integrates multiple WSL strategies with advanced optimization
    techniques, adaptive learning mechanisms, and comprehensive evaluation.
    """
    
    def __init__(self, config: UnifiedFrameworkConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.strategies = []
        self.adaptive_learning = None
        self.model_selector = None
        self.unified_framework = None
        
        # Performance optimization
        self.performance_optimizer = None
        self.batch_size_optimizer = None
        self.lr_optimizer = None
        
        # Results tracking
        self.training_history = []
        self.evaluation_results = {}
        self.performance_metrics = {}
        
        # Create experiment directory
        self.experiment_dir = Path(config.save_dir) / f"unified_experiment_{int(time.time())}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self._save_config()
    
    def setup_strategies(self, data_programming_dataset=None):
        """Setup WSL strategies based on configuration"""
        print("Setting up WSL strategies...")
        
        for i, strategy_name in enumerate(self.config.strategies):
            if strategy_name == 'data_programming':
                if data_programming_dataset is None:
                    raise ValueError("Data programming dataset required for data_programming strategy")
                
                strategy = DataProgrammingStrategy(
                    dataset=data_programming_dataset,
                    aggregation_method='weighted_vote',
                    weight=self.config.strategy_weights[i]
                )
                self.strategies.append(strategy)
                print(f"✓ Added Data Programming strategy (weight: {self.config.strategy_weights[i]})")
            
            elif strategy_name == 'noise_robust':
                strategy = NoiseRobustStrategy(
                    model_type=self.config.model_type,
                    loss_type='gce',
                    weight=self.config.strategy_weights[i]
                )
                self.strategies.append(strategy)
                print(f"✓ Added Noise Robust strategy (weight: {self.config.strategy_weights[i]})")
            
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Setup adaptive learning
        self.adaptive_learning = AdaptiveLearning(
            strategies=self.strategies,
            weights=np.array(self.config.strategy_weights)
        )
        
        # Setup model selector
        self.model_selector = ModelSelector(
            criteria=['accuracy', 'f1', 'robustness']
        )
        
        # Setup unified framework
        self.unified_framework = UnifiedFramework(
            strategies=self.strategies,
            model_selector=self.model_selector
        )
        
        print(f"✓ Setup complete with {len(self.strategies)} strategies")
    
    def setup_optimization(self):
        """Setup performance optimization components"""
        print("Setting up performance optimization...")
        
        # Performance optimizer
        opt_config = OptimizationConfig(
            use_mixed_precision=self.config.use_mixed_precision,
            use_gradient_accumulation=self.config.use_gradient_accumulation,
            accumulation_steps=self.config.accumulation_steps,
            use_early_stopping=self.config.use_early_stopping,
            patience=self.config.patience
        )
        self.performance_optimizer = PerformanceOptimizer(opt_config)
        
        # Batch size optimizer
        self.batch_size_optimizer = AdaptiveBatchSizeOptimizer(
            initial_batch_size=self.config.batch_size,
            max_batch_size=512
        )
        
        # Learning rate optimizer
        self.lr_optimizer = LearningRateOptimizer(
            initial_lr=self.config.learning_rate
        )
        
        print("✓ Performance optimization setup complete")
    
    def optimize_hyperparameters(self, train_loader, val_loader, model):
        """Optimize hyperparameters using the optimizers"""
        print("Optimizing hyperparameters...")
        
        # Optimize batch size
        optimal_batch_size = self.batch_size_optimizer.optimize_batch_size(
            model=model,
            data_loader=train_loader,
            target_memory_usage=0.8
        )
        print(f"✓ Optimal batch size: {optimal_batch_size}")
        
        # Optimize learning rate
        criterion = nn.CrossEntropyLoss()
        optimal_lr = self.lr_optimizer.find_optimal_lr(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            lr_range=(1e-6, 1e-1),
            num_steps=50
        )
        print(f"✓ Optimal learning rate: {optimal_lr:.6f}")
        
        # Update configuration
        self.config.batch_size = optimal_batch_size
        self.config.learning_rate = optimal_lr
        
        return optimal_batch_size, optimal_lr
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train the enhanced unified framework"""
        print("\n" + "="*60)
        print("TRAINING ENHANCED UNIFIED WSL FRAMEWORK")
        print("="*60)
        
        # Setup if not already done
        if not self.strategies:
            self.setup_strategies()
        if not self.performance_optimizer:
            self.setup_optimization()
        
        # Train unified framework
        print("\nTraining unified framework...")
        start_time = time.time()
        
        performances = self.unified_framework.train(
            X=X_train,
            y=y_train,
            val_X=X_val,
            val_y=y_val,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            lr=self.config.learning_rate
        )
        
        training_time = time.time() - start_time
        print(f"✓ Training completed in {training_time:.2f} seconds")
        
        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            print("\nEvaluating on test set...")
            test_metrics = self.unified_framework.evaluate(
                X=X_test,
                y=y_test,
                noise_level=0.1
            )
            self.evaluation_results['test'] = test_metrics
            print(f"✓ Test accuracy: {test_metrics['accuracy']:.4f}")
        
        # Store results
        self.evaluation_results['training'] = performances
        self.evaluation_results['training_time'] = training_time
        
        # Generate performance report
        if self.performance_optimizer:
            self.performance_metrics = self.performance_optimizer.get_performance_report()
        
        return self.evaluation_results
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """Perform cross-validation"""
        print(f"\nPerforming {n_splits}-fold cross-validation...")
        
        from sklearn.model_selection import StratifiedKFold
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = {
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Fold {fold + 1}/{n_splits}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train on this fold
            fold_results = self.train(
                X_train=X_train_fold,
                y_train=y_train_fold,
                X_val=X_val_fold,
                y_val=y_val_fold
            )
            
            # Store results
            for metric in cv_results:
                if metric in fold_results.get('test', {}):
                    cv_results[metric].append(fold_results['test'][metric])
        
        # Calculate statistics
        cv_summary = {}
        for metric, values in cv_results.items():
            cv_summary[f'{metric}_mean'] = np.mean(values)
            cv_summary[f'{metric}_std'] = np.std(values)
        
        self.evaluation_results['cross_validation'] = cv_summary
        print("✓ Cross-validation completed")
        
        return cv_summary
    
    def evaluate_robustness(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3]
    ) -> Dict[str, List[float]]:
        """Evaluate robustness to different noise levels"""
        print(f"\nEvaluating robustness to noise levels: {noise_levels}")
        
        robustness_results = {
            'noise_levels': noise_levels,
            'accuracy': [],
            'f1': []
        }
        
        for noise_level in noise_levels:
            print(f"Testing with {noise_level*100:.0f}% noise...")
            
            # Add noise to data
            noisy_X = X + np.random.normal(0, noise_level, X.shape)
            
            # Evaluate
            metrics = self.unified_framework.evaluate(
                X=noisy_X,
                y=y,
                noise_level=noise_level
            )
            
            robustness_results['accuracy'].append(metrics['accuracy'])
            robustness_results['f1'].append(metrics['f1'])
        
        self.evaluation_results['robustness'] = robustness_results
        print("✓ Robustness evaluation completed")
        
        return robustness_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if not self.config.create_visualizations:
            return
        
        print("\nCreating visualizations...")
        
        # Create plots directory
        plots_dir = self.experiment_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Strategy performance comparison
        if 'training' in self.evaluation_results:
            self._plot_strategy_performance(plots_dir)
        
        # 2. Cross-validation results
        if 'cross_validation' in self.evaluation_results:
            self._plot_cross_validation_results(plots_dir)
        
        # 3. Robustness analysis
        if 'robustness' in self.evaluation_results:
            self._plot_robustness_analysis(plots_dir)
        
        # 4. Performance metrics
        if self.performance_metrics:
            self._plot_performance_metrics(plots_dir)
        
        print(f"✓ Visualizations saved to {plots_dir}")
    
    def _plot_strategy_performance(self, plots_dir: Path):
        """Plot strategy performance comparison"""
        performances = self.evaluation_results['training']
        
        plt.figure(figsize=(10, 6))
        strategies = list(performances.keys())
        accuracies = list(performances.values())
        
        bars = plt.bar(strategies, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('Strategy Performance Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Strategy', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cross_validation_results(self, plots_dir: Path):
        """Plot cross-validation results"""
        cv_results = self.evaluation_results['cross_validation']
        
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        means = [cv_results[f'{m}_mean'] for m in metrics]
        stds = [cv_results[f'{m}_std'] for m in metrics]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, means, yerr=stds, capsize=5, 
                      color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.title('Cross-Validation Results', fontsize=14, fontweight='bold')
        plt.xlabel('Metric', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'cross_validation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_robustness_analysis(self, plots_dir: Path):
        """Plot robustness analysis"""
        robustness = self.evaluation_results['robustness']
        
        plt.figure(figsize=(10, 6))
        noise_levels = [n * 100 for n in robustness['noise_levels']]
        
        plt.plot(noise_levels, robustness['accuracy'], 'o-', label='Accuracy', linewidth=2, markersize=8)
        plt.plot(noise_levels, robustness['f1'], 's-', label='F1 Score', linewidth=2, markersize=8)
        
        plt.title('Robustness to Label Noise', fontsize=14, fontweight='bold')
        plt.xlabel('Noise Level (%)', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'robustness_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_metrics(self, plots_dir: Path):
        """Plot performance metrics over time"""
        if not self.performance_metrics:
            return
        
        # Plot training loss over time
        if self.performance_optimizer and self.performance_optimizer.performance_history:
            losses = [p['loss'] for p in self.performance_optimizer.performance_history]
            times = [p['training_time'] for p in self.performance_optimizer.performance_history]
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(losses, 'b-', linewidth=2)
            plt.title('Training Loss Over Time', fontsize=12, fontweight='bold')
            plt.xlabel('Step', fontsize=10)
            plt.ylabel('Loss', fontsize=10)
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(times, 'r-', linewidth=2)
            plt.title('Training Time Per Step', fontsize=12, fontweight='bold')
            plt.xlabel('Step', fontsize=10)
            plt.ylabel('Time (s)', fontsize=10)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_results(self):
        """Save all results and configurations"""
        if not self.config.save_results:
            return
        
        print(f"\nSaving results to {self.experiment_dir}")
        
        # Save evaluation results
        results_file = self.experiment_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        # Save performance metrics
        if self.performance_metrics:
            metrics_file = self.experiment_dir / 'performance_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2, default=str)
        
        # Save configuration
        config_file = self.experiment_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        # Save summary report
        self._save_summary_report()
        
        print("✓ Results saved successfully")
    
    def _save_summary_report(self):
        """Save a human-readable summary report"""
        report_file = self.experiment_dir / 'summary_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Enhanced Unified WSL Framework - Summary Report\n\n")
            
            f.write("## Configuration\n")
            f.write(f"- **Strategies**: {', '.join(self.config.strategies)}\n")
            f.write(f"- **Model Type**: {self.config.model_type}\n")
            f.write(f"- **Batch Size**: {self.config.batch_size}\n")
            f.write(f"- **Learning Rate**: {self.config.learning_rate}\n")
            f.write(f"- **Epochs**: {self.config.epochs}\n\n")
            
            f.write("## Results\n")
            if 'test' in self.evaluation_results:
                test_metrics = self.evaluation_results['test']
                f.write("### Test Set Performance\n")
                for metric, value in test_metrics.items():
                    f.write(f"- **{metric}**: {value:.4f}\n")
                f.write("\n")
            
            if 'cross_validation' in self.evaluation_results:
                cv_results = self.evaluation_results['cross_validation']
                f.write("### Cross-Validation Results\n")
                for metric, value in cv_results.items():
                    f.write(f"- **{metric}**: {value:.4f}\n")
                f.write("\n")
            
            if 'training_time' in self.evaluation_results:
                f.write(f"### Training Time\n")
                f.write(f"- **Total Time**: {self.evaluation_results['training_time']:.2f} seconds\n\n")
            
            f.write("## Performance Optimization\n")
            if self.performance_metrics:
                f.write(f"- **Total Steps**: {self.performance_metrics.get('total_steps', 'N/A')}\n")
                f.write(f"- **Average Training Time**: {self.performance_metrics.get('average_training_time', 'N/A'):.4f} seconds\n")
                f.write(f"- **Peak Memory Usage**: {self.performance_metrics.get('peak_memory_usage', 'N/A'):.2f} MB\n")
    
    def _save_config(self):
        """Save configuration to experiment directory"""
        config_file = self.experiment_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment"""
        summary = {
            'experiment_dir': str(self.experiment_dir),
            'config': asdict(self.config),
            'results': self.evaluation_results,
            'performance': self.performance_metrics
        }
        return summary 