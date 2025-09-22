import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

def calculate_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_prob: torch.Tensor = None
) -> Dict[str, float]:
    """
    Calculate various classification metrics
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
    Returns:
        Dictionary of metric names and values
    """
    # Convert tensors to numpy arrays
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    if y_prob is not None:
        y_prob = y_prob.cpu().numpy()
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    # Add ROC AUC and PR AUC if probabilities are provided
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        except:
            pass
    
    return metrics

def calculate_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor
) -> np.ndarray:
    """
    Calculate confusion matrix
    Args:
        y_true: True labels
        y_pred: Predicted labels
    Returns:
        Confusion matrix
    """
    return confusion_matrix(
        y_true.cpu().numpy(),
        y_pred.cpu().numpy()
    )

def calculate_class_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_prob: torch.Tensor = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for each class
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
    Returns:
        Dictionary of class metrics
    """
    # Convert tensors to numpy arrays
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    if y_prob is not None:
        y_prob = y_prob.cpu().numpy()
    
    # Get unique classes
    classes = np.unique(y_true)
    
    # Calculate metrics for each class
    class_metrics = {}
    for c in classes:
        # Create binary labels for current class
        y_true_binary = (y_true == c).astype(int)
        y_pred_binary = (y_pred == c).astype(int)
        
        # Calculate metrics
        metrics = {
            'precision': precision_score(y_true_binary, y_pred_binary),
            'recall': recall_score(y_true_binary, y_pred_binary),
            'f1': f1_score(y_true_binary, y_pred_binary)
        }
        
        # Add ROC AUC and PR AUC if probabilities are provided
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true_binary, y_prob[:, c])
                metrics['pr_auc'] = average_precision_score(y_true_binary, y_prob[:, c])
            except:
                pass
        
        class_metrics[f'class_{c}'] = metrics
    
    return class_metrics

def calculate_confidence_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_prob: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate confidence-based metrics
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
    Returns:
        Dictionary of confidence metrics
    """
    # Convert tensors to numpy arrays
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_prob = y_prob.cpu().numpy()
    
    # Calculate confidence scores
    confidence = np.max(y_prob, axis=1)
    correct = (y_true == y_pred).astype(float)
    
    # Calculate metrics
    metrics = {
        'mean_confidence': np.mean(confidence),
        'mean_confidence_correct': np.mean(confidence[correct == 1]),
        'mean_confidence_incorrect': np.mean(confidence[correct == 0]),
        'confidence_correlation': np.corrcoef(confidence, correct)[0, 1]
    }
    
    return metrics

def calculate_calibration_metrics(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Calculate calibration metrics
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        num_bins: Number of bins for calibration
    Returns:
        Dictionary of calibration metrics
    """
    # Convert tensors to numpy arrays
    y_true = y_true.cpu().numpy()
    y_prob = y_prob.cpu().numpy()
    
    # Get predicted class probabilities
    pred_prob = np.max(y_prob, axis=1)
    pred_class = np.argmax(y_prob, axis=1)
    
    # Calculate calibration error
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(pred_prob, bin_edges) - 1
    
    calibration_error = 0
    for i in range(num_bins):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:
            mean_prob = np.mean(pred_prob[mask])
            mean_acc = np.mean(pred_class[mask] == y_true[mask])
            calibration_error += np.abs(mean_prob - mean_acc)
    
    calibration_error /= num_bins
    
    return {
        'calibration_error': calibration_error
    }

def compute_noise_metrics(clean_labels: np.ndarray,
                         noisy_labels: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics related to label noise
    
    Args:
        clean_labels: True labels
        noisy_labels: Noisy labels
        
    Returns:
        Dictionary of noise-related metrics
    """
    noise_mask = clean_labels != noisy_labels
    noise_rate = np.mean(noise_mask)
    
    return {
        'noise_rate': noise_rate,
        'noise_entropy': -noise_rate * np.log2(noise_rate) - (1-noise_rate) * np.log2(1-noise_rate),
        'noise_correlation': np.corrcoef(clean_labels, noisy_labels)[0, 1]
    }

def compute_robustness_metrics(model: torch.nn.Module,
                             dataloader: torch.utils.data.DataLoader,
                             device: str = 'cuda') -> Dict[str, float]:
    """
    Compute model robustness metrics
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        
    Returns:
        Dictionary of robustness metrics
    """
    model.eval()
    all_preds = []
    all_clean_labels = []
    all_noisy_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, noisy_labels, clean_labels = batch
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_clean_labels.extend(clean_labels.squeeze().numpy())
            all_noisy_labels.extend(noisy_labels.squeeze().numpy())
            
    all_preds = np.array(all_preds)
    all_clean_labels = np.array(all_clean_labels)
    all_noisy_labels = np.array(all_noisy_labels)
    
    # Compute metrics
    metrics = calculate_metrics(torch.tensor(all_clean_labels), torch.tensor(all_preds))
    noise_metrics = compute_noise_metrics(all_clean_labels, all_noisy_labels)
    
    # Add robustness-specific metrics
    metrics.update({
        'clean_accuracy': metrics['accuracy'],
        'noisy_accuracy': accuracy_score(all_noisy_labels, all_preds),
        'noise_robustness': metrics['accuracy'] / noise_metrics['noise_rate'] if noise_metrics['noise_rate'] > 0 else 1.0
    })
    
    return metrics 