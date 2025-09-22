# WSL Framework Experiment Execution Commands
## Complete Command List with Epoch Numbers for Report and Paper

This file contains all the execution commands with specific epoch numbers for the Weakly Supervised Learning Framework experiments. Use these commands to reproduce all results for your report and paper.

---

## 📊 **Table 1: Baseline Experiments (Traditional Supervised Learning)**

### **CIFAR-10 Baseline Experiments**

```bash
# 1. CNN CIFAR-10 Traditional (Baseline: 82.1%)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --epochs 100 \
    --batch_size 128 \
    --noise_rate 0.0 \
    --lr 0.001

# 2. ResNet18 CIFAR-10 Traditional (Baseline: 80.05%)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type resnet \
    --epochs 100 \
    --batch_size 256 \
    --noise_rate 0.0 \
    --lr 0.001

# 3. MLP MNIST Traditional (Baseline: 98.17%)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/run_experiment.py \
    --dataset mnist \
    --model_type mlp \
    --epochs 50 \
    --batch_size 128 \
    --noise_rate 0.0 \
    --lr 0.001
```

---

## 🔧 **Table 2: Robust Model Experiments (Noise-Robust Training)**

### **CIFAR-10 Robust Experiments**

```bash
# 4. CNN CIFAR-10 Robust CNN (GCE Loss)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type robust_cnn \
    --epochs 100 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --loss_type gce \
    --lr 0.001

# 5. CNN CIFAR-10 Robust CNN (SCE Loss)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type robust_cnn \
    --epochs 100 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --loss_type sce \
    --lr 0.001

# 6. ResNet18 CIFAR-10 Robust ResNet18 (GCE Loss)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type robust_resnet \
    --epochs 100 \
    --batch_size 256 \
    --noise_rate 0.1 \
    --loss_type gce \
    --lr 0.001

# 7. ResNet18 CIFAR-10 Robust ResNet18 (SCE Loss)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type robust_resnet \
    --epochs 100 \
    --batch_size 256 \
    --noise_rate 0.1 \
    --loss_type sce \
    --lr 0.001
```

### **MNIST Robust Experiments**

```bash
# 8. MLP MNIST Robust MLP (GCE Loss)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/run_experiment.py \
    --dataset mnist \
    --model_type robust_mlp \
    --epochs 50 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --loss_type gce \
    --lr 0.001

# 9. MLP MNIST Robust MLP (SCE Loss)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/run_experiment.py \
    --dataset mnist \
    --model_type robust_mlp \
    --epochs 50 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --loss_type sce \
    --lr 0.001
```

---

## 🎯 **Table 3: Semi-Supervised Learning Experiments**

### **CIFAR-10 Semi-Supervised Experiments**

```bash
# 10. CNN CIFAR-10 Consistency Regularization
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/semi_supervised_experiments.py --dataset cifar10 --model_type simple_cnn --labeled_ratio 0.1 --strategy consistency --epochs 100 --batch_size 128 --learning_rate 0.001

# 11. CNN CIFAR-10 Pseudo-Labeling
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/semi_supervised_experiments.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --labeled_ratio 0.1 \
    --strategy pseudo_label \
    --epochs 100 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --confidence_threshold 0.95

# 12. CNN CIFAR-10 Co-Training
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/semi_supervised_experiments.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --labeled_ratio 0.1 \
    --strategy co_training \
    --epochs 100 \
    --batch_size 128 \
    --learning_rate 0.001

# 13. ResNet18 CIFAR-10 Consistency Regularization
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/semi_supervised_experiments.py \
    --dataset cifar10 \
    --model_type resnet \
    --labeled_ratio 0.1 \
    --strategy consistency \
    --epochs 100 \
    --batch_size 256 \
    --learning_rate 0.001

# 14. ResNet18 CIFAR-10 Pseudo-Labeling
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/semi_supervised_experiments.py \
    --dataset cifar10 \
    --model_type resnet \
    --labeled_ratio 0.1 \
    --strategy pseudo_label \
    --epochs 100 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --confidence_threshold 0.95

# 15. ResNet18 CIFAR-10 Co-Training
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/semi_supervised_experiments.py \
    --dataset cifar10 \
    --model_type resnet \
    --labeled_ratio 0.1 \
    --strategy co_training \
    --epochs 100 \
    --batch_size 256 \
    --learning_rate 0.001
```

### **MNIST Semi-Supervised Experiments**

```bash
# 16. MLP MNIST Consistency Regularization
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/semi_supervised_experiments.py \
    --dataset mnist \
    --model_type mlp \
    --labeled_ratio 0.1 \
    --strategy consistency \
    --epochs 50 \
    --batch_size 128 \
    --learning_rate 0.001

# 17. MLP MNIST Pseudo-Labeling
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/semi_supervised_experiments.py \
    --dataset mnist \
    --model_type mlp \
    --labeled_ratio 0.1 \
    --strategy pseudo_label \
    --epochs 50 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --confidence_threshold 0.95

# 18. MLP MNIST Co-Training
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/semi_supervised_experiments.py \
    --dataset mnist \
    --model_type mlp \
    --labeled_ratio 0.1 \
    --strategy co_training \
    --epochs 50 \
    --batch_size 128 \
    --learning_rate 0.001
```

---

## 🚀 **Table 4: Unified WSL Experiments (Combined Strategies)**

### **CIFAR-10 Unified WSL Experiments**

```bash
# 19. CNN CIFAR-10 Combined WSL (All Strategies)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/unified_wsl_experiment.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --strategies consistency pseudo_label co_training \
    --epochs 100 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --learning_rate 0.0001

# 20. ResNet18 CIFAR-10 Combined WSL (All Strategies)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/unified_wsl_experiment.py \
    --dataset cifar10 \
    --model_type resnet \
    --strategies consistency pseudo_label co_training \
    --epochs 100 \
    --batch_size 256 \
    --noise_rate 0.1 \
    --learning_rate 0.0001
```

### **MNIST Unified WSL Experiments**

```bash
# 21. MLP MNIST Combined WSL (All Strategies)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/unified_wsl_experiment.py \
    --dataset mnist \
    --model_type mlp \
    --strategies consistency pseudo_label co_training \
    --epochs 50 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --learning_rate 0.0001
```

---

## 🔬 **Table 5: Noise Robustness Analysis Experiments**

### **CIFAR-10 Noise Robustness Tests**

```bash
# 22. CIFAR-10 GCE Loss Noise Robustness (0%, 10%, 20% noise)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/noise_robustness_test.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --loss_type gce \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 50 \
    --batch_size 128

# 23. CIFAR-10 SCE Loss Noise Robustness (0%, 10%, 20% noise)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/noise_robustness_test.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --loss_type sce \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 50 \
    --batch_size 128

# 24. CIFAR-10 Forward Correction Noise Robustness (0%, 10%, 20% noise)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/noise_robustness_test.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --loss_type forward_correction \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 50 \
    --batch_size 128
```

### **MNIST Noise Robustness Tests**

```bash
# 25. MNIST GCE Loss Noise Robustness (0%, 10%, 20% noise)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/noise_robustness_test.py \
    --dataset mnist \
    --model_type mlp \
    --loss_type gce \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 30 \
    --batch_size 128

# 26. MNIST SCE Loss Noise Robustness (0%, 10%, 20% noise)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/noise_robustness_test.py \
    --dataset mnist \
    --model_type mlp \
    --loss_type sce \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 30 \
    --batch_size 128

# 27. MNIST Forward Correction Noise Robustness (0%, 10%, 20% noise)
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/noise_robustness_test.py \
    --dataset mnist \
    --model_type mlp \
    --loss_type forward_correction \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 30 \
    --batch_size 128
```

---

## 📈 **Table 6: Analysis and Evaluation Experiments**

### **Performance Analysis**

```bash
# 28. Generate Comprehensive Performance Report
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/generate_performance_report.py \
    --datasets cifar10 mnist \
    --model_types simple_cnn robust_cnn resnet robust_resnet mlp robust_mlp \
    --strategies traditional consistency pseudo_label co_training combined \
    --output_file performance_comparison_report.json

# 29. Generate Training Curves Visualization
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/visualization/generate_training_curves.py \
    --datasets cifar10 mnist \
    --strategies traditional consistency pseudo_label co_training combined \
    --epochs 100 \
    --output_dir ./training_curves

# 30. Generate Confusion Matrices
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/generate_confusion_matrices.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --strategy combined \
    --output_file cifar10_confusion_matrix.png

export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/generate_confusion_matrices.py \
    --dataset mnist \
    --model_type robust_mlp \
    --strategy combined \
    --output_file mnist_confusion_matrix.png
```

### **Feature Engineering Analysis**

```bash
# 31. Feature Engineering Analysis
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/feature_engineering_analysis.py \
    --strategies consistency pseudo_label co_training combined \
    --datasets cifar10 mnist \
    --output_file feature_engineering_results.json

# 32. Generate Feature Engineering Plots
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/visualization/feature_engineering_plots.py \
    --input_file feature_engineering_results.json \
    --output_dir ./feature_engineering_plots
```

### **Data Augmentation Analysis**

```bash
# 33. CIFAR-10 Data Augmentation Analysis
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/data_augmentation_analysis.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --augmentations random_rotation horizontal_flip random_crop color_jitter \
    --epochs 50 \
    --batch_size 128

# 34. MNIST Data Augmentation Analysis
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/data_augmentation_analysis.py \
    --dataset mnist \
    --model_type mlp \
    --augmentations random_rotation gaussian_noise \
    --epochs 30 \
    --batch_size 128

# 35. Generate Augmentation Comparison Plots
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/visualization/augmentation_plots.py \
    --output_dir ./augmentation_plots
```

### **Hardware and Resource Analysis**

```bash
# 36. Hardware Configuration Testing
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/hardware_configuration_test.py \
    --cpu_test \
    --gpu_test \
    --memory_test \
    --storage_test \
    --output_file hardware_test_results.json

# 37. Memory Usage Analysis
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/memory_analysis.py \
    --datasets cifar10 mnist \
    --model_types simple_cnn robust_cnn resnet robust_resnet mlp robust_mlp \
    --batch_sizes 32 64 128 256

# 38. Training Time Analysis
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/training_time_analysis.py \
    --datasets cifar10 mnist \
    --model_types simple_cnn robust_cnn resnet robust_resnet mlp robust_mlp \
    --epochs 20
```

### **Dataset and Model Analysis**

```bash
# 39. Dataset Quality Analysis
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/dataset_quality_analysis.py \
    --datasets cifar10 mnist \
    --metrics completeness relevance consistency diversity \
    --output_file dataset_quality_results.json

# 40. Model Architecture Analysis
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/model_architecture_analysis.py \
    --models simple_cnn robust_cnn resnet robust_resnet mlp robust_mlp \
    --datasets cifar10 mnist \
    --output_file model_architecture_results.json

# 41. Generate Architecture Comparison Plots
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/visualization/model_architecture_plots.py \
    --input_file model_architecture_results.json \
    --output_dir ./architecture_plots
```

---

## 🧪 **Table 7: Comprehensive Testing Suite**

```bash
# 42. Comprehensive Testing Suite
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/tests/run_comprehensive_tests.py \
    --test_modules data_preprocessing strategy_selection model_training evaluation \
    --test_types unit integration system performance \
    --output_file comprehensive_test_results.json

# 43. Generate Test Results Plots
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/visualization/test_results_plots.py \
    --input_file comprehensive_test_results.json \
    --output_dir ./test_results_plots
```

---

## 📊 **Table 8: Final Report Generation**

```bash
# 44. Generate Comprehensive Final Report
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/generate_comprehensive_report.py \
    --include_performance \
    --include_feature_engineering \
    --include_data_augmentation \
    --include_hardware_analysis \
    --include_testing_results \
    --output_file comprehensive_final_report.md

# 45. Generate Executive Summary
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/generate_executive_summary.py \
    --input_file comprehensive_final_report.md \
    --output_file executive_summary.md

# 46. Generate Summary Report
export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python src/experiments/generate_summary_report.py \
    --output_file final_summary_report.md
```

---

## 📋 **Epoch Number Summary Table**

| Experiment Category | Model Type | Dataset | Strategy | Epochs | Reason |
|-------------------|------------|---------|----------|--------|---------|
| **Baseline** | CNN | CIFAR-10 | Traditional | 100 | Full convergence for complex dataset |
| **Baseline** | ResNet18 | CIFAR-10 | Traditional | 100 | Deep architecture needs more epochs |
| **Baseline** | MLP | MNIST | Traditional | 50 | Simpler dataset, faster convergence |
| **Robust** | CNN/ResNet | CIFAR-10 | GCE/SCE | 100 | Noise-robust training needs more epochs |
| **Robust** | MLP | MNIST | GCE/SCE | 50 | Simpler dataset with noise handling |
| **Semi-Supervised** | CNN/ResNet | CIFAR-10 | All Strategies | 100 | WSL strategies need more epochs |
| **Semi-Supervised** | MLP | MNIST | All Strategies | 50 | Faster convergence on simple dataset |
| **Unified WSL** | CNN/ResNet | CIFAR-10 | Combined | 100 | Multiple strategies need more epochs |
| **Unified WSL** | MLP | MNIST | Combined | 50 | Efficient convergence with combined strategies |
| **Noise Analysis** | CNN | CIFAR-10 | All Loss Types | 50 | Sufficient for noise robustness testing |
| **Noise Analysis** | MLP | MNIST | All Loss Types | 30 | Faster convergence for noise testing |
| **Analysis** | All | All | All | 20-50 | Sufficient for analysis and comparison |

---

## 🎯 **Quick Execution Guide**

### **For Report/Paper Results (Priority Order):**

1. **Start with baseline experiments** (Commands 1-3)
2. **Run robust model experiments** (Commands 4-9)
3. **Execute semi-supervised experiments** (Commands 10-18)
4. **Run unified WSL experiments** (Commands 19-21)
5. **Perform noise robustness analysis** (Commands 22-27)
6. **Generate analysis and reports** (Commands 28-46)

### **Expected Execution Times:**

- **Baseline Experiments**: 30-750 minutes per experiment
- **Robust Experiments**: 30-450 minutes per experiment
- **Semi-Supervised Experiments**: 35-68 minutes per experiment
- **Unified WSL Experiments**: 62-450 minutes per experiment
- **Analysis Experiments**: 5-30 minutes per experiment

### **Total Estimated Time:**
- **Complete Suite**: ~48-72 hours (2-3 days)
- **Core Experiments**: ~24-36 hours (1-1.5 days)
- **Quick Demo**: ~8-12 hours (select key experiments)

---

## 📝 **Notes for Report/Paper:**

1. **All commands include proper epoch numbers** based on model complexity and dataset
2. **CIFAR-10 experiments use 100 epochs** for deep architectures and complex data
3. **MNIST experiments use 30-50 epochs** for faster convergence on simpler data
4. **Unified WSL experiments use 100 epochs** for combined strategies
5. **Noise analysis uses 30-50 epochs** for efficient testing
6. **All commands are ready for execution** with proper environment setup

**Use these commands to generate all results for your comprehensive report and paper!** 🚀 