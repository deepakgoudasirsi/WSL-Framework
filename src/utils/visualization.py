import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any

def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Path = None
) -> None:
    """
    Plot training curves for loss and accuracy
    Args:
        metrics: Dictionary containing training metrics
        save_path: Path to save the plot
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(metrics['train_loss'], label='Train Loss')
    ax1.plot(metrics['val_loss'], label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(metrics['train_acc'], label='Train Accuracy')
    ax2.plot(metrics['val_acc'], label='Val Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    # Save plot if path is provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_confusion_matrix(
    confusion_matrix: Any,
    class_names: List[str],
    save_path: Path = None
) -> None:
    """
    Plot confusion matrix
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_feature_importance(
    feature_importance: Dict[str, float],
    save_path: Path = None
) -> None:
    """
    Plot feature importance
    Args:
        feature_importance: Dictionary of feature names and their importance scores
        save_path: Path to save the plot
    """
    # Sort features by importance
    features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.bar(
        [f[0] for f in features],
        [f[1] for f in features]
    )
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_learning_curves(
    train_sizes: List[int],
    train_scores: List[float],
    val_scores: List[float],
    save_path: Path = None
) -> None:
    """
    Plot learning curves
    Args:
        train_sizes: List of training set sizes
        train_scores: List of training scores
        val_scores: List of validation scores
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        train_sizes,
        train_scores,
        'o-',
        label='Training Score'
    )
    plt.plot(
        train_sizes,
        val_scores,
        'o-',
        label='Validation Score'
    )
    plt.title('Learning Curves')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend()
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show() 