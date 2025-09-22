import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Metrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.confidences = []
        
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, confidences: torch.Tensor = None):
        """Update metrics with new batch of predictions and targets."""
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        if confidences is not None:
            self.confidences.extend(confidences.cpu().numpy())
            
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics."""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='macro'),
            'recall': recall_score(targets, predictions, average='macro'),
            'f1': f1_score(targets, predictions, average='macro')
        }
        
        if len(self.confidences) > 0:
            metrics['confidence'] = np.mean(self.confidences)
            
        return metrics
    
    def plot_confusion_matrix(self, save_path: str = None):
        """Plot and optionally save confusion matrix."""
        cm = confusion_matrix(self.targets, self.predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def compute_noise_tolerance(self, clean_predictions: List[int]) -> float:
        """Compute noise tolerance metric."""
        clean_predictions = np.array(clean_predictions)
        noisy_predictions = np.array(self.predictions)
        return np.mean(clean_predictions == noisy_predictions)
    
    def compute_stability_score(self, predictions_list: List[List[int]]) -> float:
        """Compute stability score across multiple predictions."""
        predictions_array = np.array(predictions_list)
        stability = np.mean(np.std(predictions_array, axis=0) == 0)
        return stability

def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    metrics: Metrics
) -> Dict[str, float]:
    """Evaluate model on given dataloader."""
    model.eval()
    metrics.reset()
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            confidences = torch.max(torch.softmax(outputs, dim=1), dim=1)[0]
            
            metrics.update(predictions, targets, confidences)
            
    return metrics.compute_metrics()

def plot_learning_curves(
    train_metrics: List[Dict[str, float]],
    val_metrics: List[Dict[str, float]],
    save_path: str = None
):
    """Plot learning curves for training and validation metrics."""
    epochs = range(1, len(train_metrics) + 1)
    
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(epochs, [m['accuracy'] for m in train_metrics], label='Train')
    plt.plot(epochs, [m['accuracy'] for m in val_metrics], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [m['loss'] for m in train_metrics], label='Train')
    plt.plot(epochs, [m['loss'] for m in val_metrics], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [m['f1'] for m in train_metrics], label='Train')
    plt.plot(epochs, [m['f1'] for m in val_metrics], label='Validation')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    # Example usage
    metrics = Metrics(num_classes=10)
    
    # Simulate some predictions and targets
    predictions = torch.randint(0, 10, (100,))
    targets = torch.randint(0, 10, (100,))
    confidences = torch.rand(100)
    
    metrics.update(predictions, targets, confidences)
    results = metrics.compute_metrics()
    print(results)
    
    metrics.plot_confusion_matrix() 