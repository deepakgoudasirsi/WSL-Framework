# Weakly Supervised Learning Framework

A comprehensive framework for implementing weakly supervised learning techniques using deep learning approaches.

## Overview

This project implements a unified weakly supervised learning (WSL) framework that combines multiple strategies to achieve high performance with limited labeled data. The framework supports various deep learning models and WSL strategies including consistency regularization, pseudo-labeling, and co-training.

## Project Structure

```
WSL/
├── MAJOR_PROJECT_REPORT.md    # Complete project report
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── setup.py                  # Package setup
├── src/                      # Source code
│   ├── data/                 # Data loading and preprocessing
│   ├── models/               # Model architectures
│   ├── training/             # Training utilities
│   ├── evaluation/           # Evaluation metrics
│   ├── experiments/          # Experiment runners
│   ├── utils/                # Utility functions
│   └── unified_framework/    # Unified WSL framework
└── results/                  # Generated results and figures
    ├── cifar10_confusion_matrix.png
    ├── cifar10_training_history.png
    ├── mnist_confusion_matrix.png
    ├── mnist_training_history.png
    ├── performance_comparison.png
    └── performance_results.png
```

## Key Features

- **Multiple WSL Strategies**: Consistency regularization, pseudo-labeling, co-training
- **Deep Learning Models**: CNN, ResNet, MLP architectures
- **Noise-Robust Training**: GCE, SCE, and Forward Correction loss functions
- **Comprehensive Evaluation**: Accuracy, F1-score, confusion matrices
- **Modular Design**: Easy to extend and customize

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The framework can be used through the main training script:

```bash
python src/train.py --dataset cifar10 --strategy combined --labeled_ratio 0.1
```

## Results

The framework achieves state-of-the-art performance:
- **MNIST**: 98.7% accuracy with 10% labeled data
- **CIFAR-10**: 89.3% accuracy with 10% labeled data


## License

MIT License - see LICENSE file for details. 
