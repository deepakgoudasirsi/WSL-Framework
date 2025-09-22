#!/bin/bash

# =============================================================================
# WSL Framework - Complete Server Experiment Commands
# Updated for GPU Server Execution
# =============================================================================

# Set up environment variables for the server
export PYTHONPATH=/media/rvcse22/CSERV/WSL_project:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=""  # Force CPU for now, change to GPU number when available

# Navigate to project directory
cd /media/rvcse22/CSERV/WSL_project

echo "🚀 Starting WSL Framework Experiments on Server"
echo "📁 Working Directory: $(pwd)"
echo "🔧 Environment: PYTHONPATH=$PYTHONPATH"
echo "💻 Device: CPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo ""

# =============================================================================
# TABLE 1: BASELINE EXPERIMENTS (Traditional Supervised Learning)
# =============================================================================

echo "📊 TABLE 1: Baseline Experiments (Traditional Supervised Learning)"
echo "=================================================================="

# 1. CNN CIFAR-10 Traditional (Baseline: 82.1%)
echo "Running: CNN CIFAR-10 Traditional (Baseline: 82.1%)"
python3 src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --epochs 100 \
    --batch_size 128 \
    --noise_rate 0.0 \
    --lr 0.001

# 2. ResNet18 CIFAR-10 Traditional (Baseline: 80.05%)
echo "Running: ResNet18 CIFAR-10 Traditional (Baseline: 80.05%)"
python3 src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type resnet \
    --epochs 100 \
    --batch_size 256 \
    --noise_rate 0.0 \
    --lr 0.001

# 3. MLP MNIST Traditional (Baseline: 98.17%)
echo "Running: MLP MNIST Traditional (Baseline: 98.17%)"
python3 src/experiments/run_experiment.py \
    --dataset mnist \
    --model_type mlp \
    --epochs 100 \
    --batch_size 128 \
    --noise_rate 0.0 \
    --lr 0.001

# =============================================================================
# TABLE 2: ROBUST MODEL EXPERIMENTS (Noise-Robust Training)
# =============================================================================

echo ""
echo "🔧 TABLE 2: Robust Model Experiments (Noise-Robust Training)"
echo "============================================================"

# 4. CNN CIFAR-10 Robust CNN (GCE Loss)
echo "Running: CNN CIFAR-10 Robust CNN (GCE Loss)"
python3 src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type robust_cnn \
    --epochs 100 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --loss_type gce \
    --lr 0.001

# 5. CNN CIFAR-10 Robust CNN (SCE Loss)
echo "Running: CNN CIFAR-10 Robust CNN (SCE Loss)"
python3 src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type robust_cnn \
    --epochs 100 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --loss_type sce \
    --lr 0.001

# 6. ResNet18 CIFAR-10 Robust ResNet18 (GCE Loss)
echo "Running: ResNet18 CIFAR-10 Robust ResNet18 (GCE Loss)"
python3 src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type robust_resnet \
    --epochs 100 \
    --batch_size 256 \
    --noise_rate 0.1 \
    --loss_type gce \
    --lr 0.001

# 7. ResNet18 CIFAR-10 Robust ResNet18 (SCE Loss)
echo "Running: ResNet18 CIFAR-10 Robust ResNet18 (SCE Loss)"
python3 src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type robust_resnet \
    --epochs 100 \
    --batch_size 256 \
    --noise_rate 0.1 \
    --loss_type sce \
    --lr 0.001

# 8. MLP MNIST Robust MLP (GCE Loss)
echo "Running: MLP MNIST Robust MLP (GCE Loss)"
python3 src/experiments/run_experiment.py \
    --dataset mnist \
    --model_type robust_mlp \
    --epochs 50 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --loss_type gce \
    --lr 0.001

# 9. MLP MNIST Robust MLP (SCE Loss)
echo "Running: MLP MNIST Robust MLP (SCE Loss)"
python3 src/experiments/run_experiment.py \
    --dataset mnist \
    --model_type robust_mlp \
    --epochs 50 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --loss_type sce \
    --lr 0.001

# =============================================================================
# TABLE 3: SEMI-SUPERVISED LEARNING EXPERIMENTS
# =============================================================================

echo ""
echo "🎯 TABLE 3: Semi-Supervised Learning Experiments"
echo "================================================"

# 10. CNN CIFAR-10 Consistency Regularization
echo "Running: CNN CIFAR-10 Consistency Regularization"
python3 src/experiments/semi_supervised_experiments.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --labeled_ratio 0.1 \
    --strategy consistency \
    --epochs 100 \
    --batch_size 128 \
    --learning_rate 0.001

# 11. CNN CIFAR-10 Pseudo-Labeling
echo "Running: CNN CIFAR-10 Pseudo-Labeling"
python3 src/experiments/semi_supervised_experiments.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --labeled_ratio 0.1 \
    --strategy pseudo_label \
    --epochs 100 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --confidence_threshold 0.95

# 12. CNN CIFAR-10 Co-Training
echo "Running: CNN CIFAR-10 Co-Training"
python3 src/experiments/semi_supervised_experiments.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --labeled_ratio 0.1 \
    --strategy co_training \
    --epochs 100 \
    --batch_size 128 \
    --learning_rate 0.001

# 13. ResNet18 CIFAR-10 Consistency Regularization
echo "Running: ResNet18 CIFAR-10 Consistency Regularization"
python3 src/experiments/semi_supervised_experiments.py \
    --dataset cifar10 \
    --model_type resnet \
    --labeled_ratio 0.1 \
    --strategy consistency \
    --epochs 100 \
    --batch_size 256 \
    --learning_rate 0.001

# 14. ResNet18 CIFAR-10 Pseudo-Labeling
echo "Running: ResNet18 CIFAR-10 Pseudo-Labeling"
python3 src/experiments/semi_supervised_experiments.py \
    --dataset cifar10 \
    --model_type resnet \
    --labeled_ratio 0.1 \
    --strategy pseudo_label \
    --epochs 100 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --confidence_threshold 0.95

# 15. ResNet18 CIFAR-10 Co-Training
echo "Running: ResNet18 CIFAR-10 Co-Training"
python3 src/experiments/semi_supervised_experiments.py \
    --dataset cifar10 \
    --model_type resnet \
    --labeled_ratio 0.1 \
    --strategy co_training \
    --epochs 100 \
    --batch_size 256 \
    --learning_rate 0.001

# 16. MLP MNIST Consistency Regularization
echo "Running: MLP MNIST Consistency Regularization"
python3 src/experiments/semi_supervised_experiments.py \
    --dataset mnist \
    --model_type mlp \
    --labeled_ratio 0.1 \
    --strategy consistency \
    --epochs 50 \
    --batch_size 128 \
    --learning_rate 0.001

# 17. MLP MNIST Pseudo-Labeling
echo "Running: MLP MNIST Pseudo-Labeling"
python3 src/experiments/semi_supervised_experiments.py \
    --dataset mnist \
    --model_type mlp \
    --labeled_ratio 0.1 \
    --strategy pseudo_label \
    --epochs 50 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --confidence_threshold 0.95

# 18. MLP MNIST Co-Training
echo "Running: MLP MNIST Co-Training"
python3 src/experiments/semi_supervised_experiments.py \
    --dataset mnist \
    --model_type mlp \
    --labeled_ratio 0.1 \
    --strategy co_training \
    --epochs 50 \
    --batch_size 128 \
    --learning_rate 0.001

# =============================================================================
# TABLE 4: UNIFIED WSL EXPERIMENTS (Combined Strategies)
# =============================================================================

echo ""
echo "🚀 TABLE 4: Unified WSL Experiments (Combined Strategies)"
echo "========================================================"

# 19. CNN CIFAR-10 Combined WSL (All Strategies)
echo "Running: CNN CIFAR-10 Combined WSL (All Strategies)"
python3 src/experiments/unified_wsl_experiment.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --strategies consistency pseudo_label co_training \
    --epochs 100 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --learning_rate 0.0001

# 20. ResNet18 CIFAR-10 Combined WSL (All Strategies)
echo "Running: ResNet18 CIFAR-10 Combined WSL (All Strategies)"
python3 src/experiments/unified_wsl_experiment.py \
    --dataset cifar10 \
    --model_type resnet \
    --strategies consistency pseudo_label co_training \
    --epochs 100 \
    --batch_size 256 \
    --noise_rate 0.1 \
    --learning_rate 0.0001

# 21. MLP MNIST Combined WSL (All Strategies)
echo "Running: MLP MNIST Combined WSL (All Strategies)"
python3 src/experiments/unified_wsl_experiment.py \
    --dataset mnist \
    --model_type mlp \
    --strategies consistency pseudo_label co_training \
    --epochs 50 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --learning_rate 0.0001

# =============================================================================
# TABLE 5: NOISE ROBUSTNESS ANALYSIS EXPERIMENTS
# =============================================================================

echo ""
echo "🔬 TABLE 5: Noise Robustness Analysis Experiments"
echo "================================================"

# 22. CIFAR-10 GCE Loss Noise Robustness (0%, 10%, 20% noise)
echo "Running: CIFAR-10 GCE Loss Noise Robustness (0%, 10%, 20% noise)"
python3 src/experiments/noise_robustness_test.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --loss_type gce \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 50 \
    --batch_size 128

# 23. CIFAR-10 SCE Loss Noise Robustness (0%, 10%, 20% noise)
echo "Running: CIFAR-10 SCE Loss Noise Robustness (0%, 10%, 20% noise)"
python3 src/experiments/noise_robustness_test.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --loss_type sce \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 50 \
    --batch_size 128

# 24. CIFAR-10 Forward Correction Noise Robustness (0%, 10%, 20% noise)
echo "Running: CIFAR-10 Forward Correction Noise Robustness (0%, 10%, 20% noise)"
python3 src/experiments/noise_robustness_test.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --loss_type forward_correction \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 50 \
    --batch_size 128

# 25. MNIST GCE Loss Noise Robustness (0%, 10%, 20% noise)
echo "Running: MNIST GCE Loss Noise Robustness (0%, 10%, 20% noise)"
python3 src/experiments/noise_robustness_test.py \
    --dataset mnist \
    --model_type mlp \
    --loss_type gce \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 30 \
    --batch_size 128

# 26. MNIST SCE Loss Noise Robustness (0%, 10%, 20% noise)
echo "Running: MNIST SCE Loss Noise Robustness (0%, 10%, 20% noise)"
python3 src/experiments/noise_robustness_test.py \
    --dataset mnist \
    --model_type mlp \
    --loss_type sce \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 30 \
    --batch_size 128

# 27. MNIST Forward Correction Noise Robustness (0%, 10%, 20% noise)
echo "Running: MNIST Forward Correction Noise Robustness (0%, 10%, 20% noise)"
python3 src/experiments/noise_robustness_test.py \
    --dataset mnist \
    --model_type mlp \
    --loss_type forward_correction \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 30 \
    --batch_size 128

# =============================================================================
# TABLE 6: ANALYSIS AND EVALUATION EXPERIMENTS
# =============================================================================

echo ""
echo "📈 TABLE 6: Analysis and Evaluation Experiments"
echo "=============================================="

# 28. Generate Comprehensive Performance Report
echo "Running: Generate Comprehensive Performance Report"
python3 src/experiments/generate_performance_report.py \
    --datasets cifar10 mnist \
    --model_types simple_cnn robust_cnn resnet robust_resnet mlp robust_mlp \
    --strategies traditional consistency pseudo_label co_training combined \
    --output_file performance_comparison_report.json

# 29. Generate Training Curves Visualization
echo "Running: Generate Training Curves Visualization"
python3 src/visualization/generate_training_curves.py \
    --datasets cifar10 mnist \
    --strategies traditional consistency pseudo_label co_training combined \
    --epochs 100 \
    --output_dir ./training_curves

# 30. Generate Confusion Matrices
echo "Running: Generate Confusion Matrices"
python3 src/experiments/generate_confusion_matrices.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --strategy combined \
    --output_file cifar10_confusion_matrix.png

python3 src/experiments/generate_confusion_matrices.py \
    --dataset mnist \
    --model_type robust_mlp \
    --strategy combined \
    --output_file mnist_confusion_matrix.png

# =============================================================================
# TABLE 7: COMPREHENSIVE TESTING SUITE
# =============================================================================

echo ""
echo "🧪 TABLE 7: Comprehensive Testing Suite"
echo "======================================"

# 31. Comprehensive Testing Suite
echo "Running: Comprehensive Testing Suite"
python3 src/tests/run_comprehensive_tests.py \
    --test_modules data_preprocessing strategy_selection model_training evaluation \
    --test_types unit integration system performance \
    --output_file comprehensive_test_results.json

# 32. Generate Test Results Plots
echo "Running: Generate Test Results Plots"
python3 src/visualization/test_results_plots.py \
    --input_file comprehensive_test_results.json \
    --output_dir ./test_results_plots

# =============================================================================
# TABLE 8: FINAL REPORT GENERATION
# =============================================================================

echo ""
echo "📊 TABLE 8: Final Report Generation"
echo "=================================="

# 33. Generate Comprehensive Final Report
echo "Running: Generate Comprehensive Final Report"
python3 src/experiments/generate_comprehensive_report.py \
    --include_performance \
    --include_feature_engineering \
    --include_data_augmentation \
    --include_hardware_analysis \
    --include_testing_results \
    --output_file comprehensive_final_report.md

# 34. Generate Executive Summary
echo "Running: Generate Executive Summary"
python3 src/experiments/generate_executive_summary.py \
    --input_file comprehensive_final_report.md \
    --output_file executive_summary.md

# 35. Generate Summary Report
echo "Running: Generate Summary Report"
python3 src/experiments/generate_summary_report.py \
    --output_file final_summary_report.md

# =============================================================================
# COMPLETION MESSAGE
# =============================================================================

echo ""
echo "🎉 All WSL Framework Experiments Completed!"
echo "📁 Results saved in: /media/rvcse22/CSERV/WSL_project"
echo "📊 Reports generated:"
echo "   - comprehensive_final_report.md"
echo "   - executive_summary.md"
echo "   - final_summary_report.md"
echo "   - performance_comparison_report.json"
echo "   - comprehensive_test_results.json"
echo ""
echo "🚀 WSL Framework Server Execution Complete!"
echo ""

# =============================================================================
# GPU USAGE COMMANDS (For Reference)
# =============================================================================

echo "💡 GPU Usage Commands (for future reference):"
echo "=============================================="
echo "# Use GPU 1 (T400) - less busy"
echo "export CUDA_VISIBLE_DEVICES=1"
echo ""
echo "# Use GPU 0 or 2 (A100) - when available"
echo "export CUDA_VISIBLE_DEVICES=0"
echo "export CUDA_VISIBLE_DEVICES=2"
echo ""
echo "# Check GPU status"
echo "nvidia-smi"
echo ""
echo "# Monitor GPU usage"
echo "watch -n 10 nvidia-smi"
echo "" 