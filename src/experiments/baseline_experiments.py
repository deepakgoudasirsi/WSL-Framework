import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from time import time
from sklearn.metrics import confusion_matrix, classification_report

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wsl_framework.framework import WSLFramework
from wsl_framework.data_loader import DataLoader

def run_baseline_experiment(dataset_name, strategy, model_type, labeled_ratio=0.1, epochs=5):
    """Run a single baseline experiment with specified parameters."""
    print(f"\nRunning experiment for {dataset_name} with {strategy} strategy...")
    
    # Load data
    data_loader = DataLoader(dataset_name)
    train_data, test_data = data_loader.load_data()
    
    # Initialize and train model
    model = WSLFramework(
        strategy=strategy,
        model_type=model_type,
        labeled_ratio=labeled_ratio
    )
    
    # Train model
    start_time = time()
    history = model.train(train_data, epochs=epochs)
    training_time = time() - start_time
    
    # Evaluate model
    metrics = model.evaluate(test_data)
    
    # Get predictions for confusion matrix
    y_pred = model.predict(test_data[0])
    y_true = test_data[1]
    
    return {
        'history': history,
        'metrics': metrics,
        'training_time': training_time,
        'predictions': y_pred,
        'true_labels': y_true
    }

def plot_experiment_results(results, save_dir='experiment_results'):
    """Plot and save experiment results."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    # Accuracy subplot
    plt.subplot(1, 2, 1)
    for dataset, result in results.items():
        accuracies = [h['accuracy'] for h in result['history']]
        plt.plot(accuracies, label=f'{dataset}')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Loss subplot
    plt.subplot(1, 2, 2)
    for dataset, result in results.items():
        losses = [h['loss'] for h in result['history']]
        plt.plot(losses, label=f'{dataset}')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_curves_{timestamp}.png'))
    plt.close()
    
    # Plot confusion matrices
    for dataset, result in results.items():
        cm = confusion_matrix(result['true_labels'], result['predictions'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {dataset}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_{dataset}_{timestamp}.png'))
        plt.close()
    
    # Save metrics to text file
    with open(os.path.join(save_dir, f'metrics_{timestamp}.txt'), 'w') as f:
        f.write("=== Experiment Results ===\n\n")
        for dataset, result in results.items():
            f.write(f"\n{dataset} Results:\n")
            f.write(f"Training Time: {result['training_time']:.2f} seconds\n")
            f.write("\nMetrics:\n")
            for metric, value in result['metrics'].items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(
                result['true_labels'],
                result['predictions']
            ))
            f.write("\n" + "="*50 + "\n")

def main():
    # Experiment configurations
    experiments = [
        {
            'dataset': 'mnist',
            'strategy': 'pseudo_labeling',
            'model_type': 'cnn',
            'labeled_ratio': 0.1
        },
        {
            'dataset': 'cifar10',
            'strategy': 'pseudo_labeling',
            'model_type': 'cnn',
            'labeled_ratio': 0.1
        }
    ]
    
    # Run experiments
    results = {}
    for exp in experiments:
        result = run_baseline_experiment(
            dataset_name=exp['dataset'],
            strategy=exp['strategy'],
            model_type=exp['model_type'],
            labeled_ratio=exp['labeled_ratio']
        )
        results[exp['dataset']] = result
    
    # Plot and save results
    plot_experiment_results(results)
    
    # Print summary
    print("\n=== Experiment Summary ===")
    for dataset, result in results.items():
        print(f"\n{dataset.upper()} Results:")
        print(f"Training Time: {result['training_time']:.2f} seconds")
        print("Metrics:")
        for metric, value in result['metrics'].items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 