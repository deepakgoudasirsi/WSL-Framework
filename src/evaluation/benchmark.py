import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from src.models.unified_wsl import UnifiedWSLModel
from sklearn.metrics import confusion_matrix, classification_report

class BenchmarkEvaluator:
    """Evaluator for benchmarking WSL methods"""
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
    
    def evaluate_model(self, model: nn.Module) -> Dict[str, float]:
        """Evaluate a single model"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Handle different model types
                if isinstance(model, UnifiedWSLModel):
                    # For WSL models, use the base model for evaluation
                    outputs = model.base_model(inputs)
                else:
                    # For regular models, use forward pass
                    outputs = model(inputs)
                
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, average='weighted'),
            'recall': recall_score(all_targets, all_preds, average='weighted'),
            'f1': f1_score(all_targets, all_preds, average='weighted')
        }
        
        return metrics
    
    def compare_methods(
        self,
        models: Dict[str, nn.Module],
        dataset_name: str
    ) -> pd.DataFrame:
        """Compare multiple methods"""
        results = []
        
        for name, model in tqdm(models.items(), desc="Comparing methods"):
            print(f"\nEvaluating {name}...")
            metrics = self.evaluate_model(model)
            metrics['method'] = name
            metrics['dataset'] = dataset_name
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def plot_comparison(
        self,
        results_df: pd.DataFrame,
        metric: str,
        save_path: Optional[str] = None
    ):
        """Plot comparison of methods"""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x='method', y=metric)
        plt.title(f'{metric.capitalize()} Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def generate_report(
        self,
        results_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Generate evaluation report"""
        report = "# Model Evaluation Report\n\n"
        
        # Overall metrics
        report += "## Overall Metrics\n\n"
        report += results_df.to_markdown(index=False)
        report += "\n\n"
        
        # Best method for each metric
        report += "## Best Methods\n\n"
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            best_method = results_df.loc[results_df[metric].idxmax()]
            report += f"- Best {metric}: {best_method['method']} ({best_method[metric]:.4f})\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)

class AblationStudy:
    """Study the impact of different components"""
    def __init__(
        self,
        base_model: nn.Module,
        test_loader: DataLoader,
        device: torch.device
    ):
        self.base_model = base_model
        self.test_loader = test_loader
        self.device = device
        self.evaluator = BenchmarkEvaluator(base_model, test_loader, device)
    
    def study_components(
        self,
        components: Dict[str, object],
        component_name: str
    ) -> pd.DataFrame:
        """Study impact of different components"""
        results = []
        
        # Evaluate base model
        print("\nEvaluating base model...")
        base_metrics = self.evaluator.evaluate_model(self.base_model)
        base_metrics[component_name] = 'none'
        results.append(base_metrics)
        
        # Evaluate with each component
        for name, component in tqdm(components.items(), desc="Studying components"):
            print(f"\nEvaluating with {name}...")
            # Create model with component
            model = self._create_model_with_component(component)
            
            # Evaluate
            metrics = self.evaluator.evaluate_model(model)
            metrics[component_name] = name
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def _create_model_with_component(self, component: object) -> nn.Module:
        """Create model with a single component"""
        # This is a placeholder - implement based on your model architecture
        return self.base_model
    
    def plot_ablation(
        self,
        results_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Plot ablation study results"""
        plt.figure(figsize=(12, 6))
        
        # Plot each metric
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            sns.barplot(data=results_df, x=results_df.columns[-1], y=metric)
            plt.title(f'{metric.capitalize()}')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

class ErrorAnalysis:
    """Analyze model errors"""
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        class_names: List[str]
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
    
    def analyze_errors(self) -> Dict[str, pd.DataFrame]:
        """Analyze different types of errors"""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Analyzing errors"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Handle different model types
                if isinstance(self.model, UnifiedWSLModel):
                    # For WSL models, use the base model for evaluation
                    outputs = self.model.base_model(inputs)
                else:
                    # For regular models, use forward pass
                    outputs = self.model(inputs)
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Analyze different types of errors
        results = {}
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        results['confusion_matrix'] = pd.DataFrame(
            cm,
            index=self.class_names,
            columns=self.class_names
        )
        
        # Classification report
        report = classification_report(
            all_targets,
            all_preds,
            target_names=self.class_names,
            output_dict=True
        )
        results['classification_report'] = pd.DataFrame(report).transpose()
        
        # Error cases
        errors = all_preds != all_targets
        error_probs = all_probs[errors]
        error_preds = all_preds[errors]
        error_targets = all_targets[errors]
        
        results['error_cases'] = pd.DataFrame({
            'true_class': [self.class_names[t] for t in error_targets],
            'predicted_class': [self.class_names[p] for p in error_preds],
            'confidence': np.max(error_probs, axis=1)
        })
        
        return results
    
    def plot_error_analysis(
        self,
        results: Dict[str, pd.DataFrame],
        save_dir: Optional[str] = None
    ):
        """Plot error analysis results"""
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        if save_dir:
            plt.savefig(f'{save_dir}/confusion_matrix.png')
        plt.close()
        
        # Plot error cases
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=results['error_cases'],
            x='true_class',
            y='confidence'
        )
        plt.title('Confidence Distribution for Errors')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/error_confidence.png')
        plt.close() 