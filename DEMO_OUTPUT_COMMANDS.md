# WSL Framework - Demo Output Commands

## 🎯 PRESENTATION DEMO COMMANDS

### 📥 1. TRAINED MODEL WEIGHTS AND ARCHITECTURE

**Command to Train and Save Model:**
```bash
# Train ResNet18 on CIFAR-10 with GCE loss
python src/main.py --dataset cifar10 --model resnet --strategy pseudo_labeling --labeled_ratio 0.1 --save_model --output_dir experiments/demo_model
```

**Command to Show Model Architecture:**
```bash
# Display model architecture
python -c "
import torch
from src.models.resnet import ResNet18
model = ResNet18(num_classes=10)
print('Model Architecture:')
print(model)
print(f'Total Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

**Command to Show Saved Model Files:**
```bash
# List saved model files
ls -la experiments/demo_model/
echo "Model files saved:"
echo "- model_weights.pth (trained weights)"
echo "- model_config.json (architecture config)"
echo "- training_history.json (training metadata)"
```

---

### 📊 2. PERFORMANCE METRICS (Accuracy, F1-Score, Precision, Recall)

**Command to Run Evaluation and Show Metrics:**
```bash
# Evaluate model and display metrics
python src/evaluation/evaluate_model.py --model_path experiments/demo_model/model_weights.pth --dataset cifar10 --output_file results/metrics.json
```

**Command to Display Metrics in Table Format:**
```bash
# Show metrics in formatted table
python -c "
import json
import pandas as pd

# Load metrics
with open('results/metrics.json', 'r') as f:
    metrics = json.load(f)

# Create formatted table
df = pd.DataFrame([metrics])
print('Performance Metrics:')
print('=' * 50)
print(f'Accuracy:     {metrics[\"accuracy\"]:.4f} ({metrics[\"accuracy\"]*100:.2f}%)')
print(f'F1-Score:     {metrics[\"f1_score\"]:.4f}')
print(f'Precision:    {metrics[\"precision\"]:.4f}')
print(f'Recall:       {metrics[\"recall\"]:.4f}')
print(f'Test Loss:    {metrics[\"test_loss\"]:.4f}')
print('=' * 50)
"
```

**Command to Show Metrics Comparison:**
```bash
# Compare metrics across different strategies
python src/analysis/compare_strategies.py --results_dir experiments/ --output_file results/strategy_comparison.csv
```

---

### 📈 3. TRAINING CURVES AND LOSS PLOTS

**Command to Generate Training Curves:**
```bash
# Generate training curves
python src/visualization/plot_training_curves.py --history_file experiments/demo_model/training_history.json --output_dir plots/
```

**Command to Display Training Curves:**
```bash
# Show training curves
echo "Training Curves Generated:"
ls -la plots/
echo ""
echo "Available plots:"
echo "- training_loss.png (loss over epochs)"
echo "- training_accuracy.png (accuracy over epochs)"
echo "- validation_metrics.png (validation metrics)"
echo "- learning_rate.png (learning rate schedule)"
```

**Command to Show Real-time Training Progress:**
```bash
# Monitor training in real-time
python src/training/train_with_monitoring.py --dataset cifar10 --model resnet --strategy pseudo_labeling --labeled_ratio 0.1 --epochs 10 --monitor
```

---

### 🎯 4. CONFUSION MATRICES

**Command to Generate Confusion Matrix:**
```bash
# Generate confusion matrix
python src/evaluation/generate_confusion_matrix.py --model_path experiments/demo_model/model_weights.pth --dataset cifar10 --output_file plots/confusion_matrix.png
```

**Command to Display Confusion Matrix:**
```bash
# Show confusion matrix
echo "Confusion Matrix Generated:"
echo "File: plots/confusion_matrix.png"
echo ""
echo "Matrix shows:"
echo "- True vs Predicted class distribution"
echo "- Per-class accuracy"
echo "- Most confused class pairs"
echo "- Overall classification performance"
```

**Command to Show Confusion Matrix Statistics:**
```bash
# Display confusion matrix statistics
python -c "
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Load predictions and true labels
predictions = np.load('results/predictions.npy')
true_labels = np.load('results/true_labels.npy')

# Generate confusion matrix
cm = confusion_matrix(true_labels, predictions)

print('Confusion Matrix Statistics:')
print('=' * 40)
print(f'Total Samples: {len(true_labels)}')
print(f'Correct Predictions: {np.sum(predictions == true_labels)}')
print(f'Accuracy: {np.sum(predictions == true_labels) / len(true_labels):.4f}')
print('=' * 40)
"
```

---

### 🎯 5. MODEL PREDICTIONS AND CONFIDENCE SCORES

**Command to Generate Predictions:**
```bash
# Generate predictions and confidence scores
python src/evaluation/generate_predictions.py --model_path experiments/demo_model/model_weights.pth --dataset cifar10 --output_file results/predictions_with_confidence.json
```

**Command to Display Sample Predictions:**
```bash
# Show sample predictions
python -c "
import json
import numpy as np

# Load predictions
with open('results/predictions_with_confidence.json', 'r') as f:
    predictions = json.load(f)

print('Sample Predictions with Confidence Scores:')
print('=' * 60)
print('Sample | True Label | Predicted | Confidence | Correct')
print('-' * 60)

for i in range(min(10, len(predictions))):
    pred = predictions[i]
    correct = '✓' if pred['true_label'] == pred['predicted_label'] else '✗'
    print(f'{i+1:6d} | {pred[\"true_label\"]:10s} | {pred[\"predicted_label\"]:10s} | {pred[\"confidence\"]:10.4f} | {correct}')

print('=' * 60)
"
```

**Command to Show Confidence Distribution:**
```bash
# Analyze confidence distribution
python src/analysis/analyze_confidence.py --predictions_file results/predictions_with_confidence.json --output_file plots/confidence_distribution.png
```

---

## 🎯 COMPREHENSIVE DEMO SCRIPT

### Complete Demo Sequence:

```bash
#!/bin/bash
echo "=== WSL Framework Demo ==="
echo ""

# 1. Train model and save
echo "1. Training model and saving weights..."
python src/main.py --dataset cifar10 --model resnet --strategy pseudo_labeling --labeled_ratio 0.1 --save_model --output_dir experiments/demo_model

# 2. Show model architecture
echo ""
echo "2. Model Architecture:"
python -c "
import torch
from src.models.resnet import ResNet18
model = ResNet18(num_classes=10)
print(f'Total Parameters: {sum(p.numel() for p in model.parameters()):,}')
"

# 3. Evaluate and show metrics
echo ""
echo "3. Performance Metrics:"
python src/evaluation/evaluate_model.py --model_path experiments/demo_model/model_weights.pth --dataset cifar10 --output_file results/metrics.json

# 4. Generate training curves
echo ""
echo "4. Generating training curves..."
python src/visualization/plot_training_curves.py --history_file experiments/demo_model/training_history.json --output_dir plots/

# 5. Generate confusion matrix
echo ""
echo "5. Generating confusion matrix..."
python src/evaluation/generate_confusion_matrix.py --model_path experiments/demo_model/model_weights.pth --dataset cifar10 --output_file plots/confusion_matrix.png

# 6. Generate predictions
echo ""
echo "6. Generating predictions and confidence scores..."
python src/evaluation/generate_predictions.py --model_path experiments/demo_model/model_weights.pth --dataset cifar10 --output_file results/predictions_with_confidence.json

echo ""
echo "=== Demo Complete ==="
echo "All outputs generated successfully!"
```

---

## 📊 OUTPUT SUMMARY COMMANDS

### Show All Generated Outputs:
```bash
# List all generated outputs
echo "=== WSL Framework Outputs ==="
echo ""
echo "1. Model Files:"
ls -la experiments/demo_model/
echo ""
echo "2. Performance Metrics:"
cat results/metrics.json
echo ""
echo "3. Training Plots:"
ls -la plots/
echo ""
echo "4. Confusion Matrix:"
ls -la plots/confusion_matrix.png
echo ""
echo "5. Predictions:"
head -10 results/predictions_with_confidence.json
```

### Quick Status Check:
```bash
# Check if all outputs are generated
python -c "
import os
import json

outputs = {
    'Model Weights': 'experiments/demo_model/model_weights.pth',
    'Model Config': 'experiments/demo_model/model_config.json',
    'Training History': 'experiments/demo_model/training_history.json',
    'Performance Metrics': 'results/metrics.json',
    'Training Curves': 'plots/training_loss.png',
    'Confusion Matrix': 'plots/confusion_matrix.png',
    'Predictions': 'results/predictions_with_confidence.json'
}

print('Output Status Check:')
print('=' * 40)
for name, path in outputs.items():
    status = '✓' if os.path.exists(path) else '✗'
    print(f'{status} {name}')
print('=' * 40)
"
```

---

## 🎯 PRESENTATION TIPS

### During Demo:
1. **Start with training**: Show the model training process
2. **Display architecture**: Show model structure and parameters
3. **Show metrics**: Display performance numbers clearly
4. **Display plots**: Show training curves and confusion matrix
5. **Show predictions**: Display sample predictions with confidence

### Key Points to Emphasize:
- **Model Efficiency**: Show parameter count and training time
- **Performance**: Highlight accuracy and other metrics
- **Robustness**: Show confidence scores and error analysis
- **Comprehensive Output**: Demonstrate all output types
- **Production Ready**: Show structured and organized outputs

### Backup Commands:
```bash
# If any command fails, use these backup commands
python src/backup_demo.py --mode quick  # Quick demo with pre-trained model
python src/show_results.py --demo_mode   # Show existing results
```

---

**Use these commands during your presentation to demonstrate the comprehensive outputs of your WSL framework! 🎯** 