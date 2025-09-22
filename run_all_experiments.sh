#!/bin/bash

# WSL Framework - Complete Experiment Execution Script
# This script runs all experiments with proper epoch numbers for report and paper

echo "🚀 Starting WSL Framework Complete Experiment Suite"
echo "=================================================="
echo ""

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create results directory
mkdir -p experiment_results
mkdir -p logs

# Function to run experiment with logging
run_experiment() {
    local experiment_name="$1"
    local command="$2"
    local log_file="logs/${experiment_name}.log"
    
    echo "📊 Running: $experiment_name"
    echo "Command: $command"
    echo "Log file: $log_file"
    echo "Started at: $(date)"
    echo "----------------------------------------"
    
    # Run the command and log output
    eval "$command" 2>&1 | tee "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    echo "----------------------------------------"
    echo "Finished at: $(date)"
    echo "Exit code: $exit_code"
    echo ""
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $experiment_name completed successfully"
    else
        echo "❌ $experiment_name failed with exit code $exit_code"
    fi
    echo ""
}

# ============================================================================
# TABLE 1: BASELINE EXPERIMENTS (Traditional Supervised Learning)
# ============================================================================

echo "📊 TABLE 1: BASELINE EXPERIMENTS"
echo "================================="

# 1. CNN CIFAR-10 Traditional (Baseline: 82.1%)
run_experiment "CNN_CIFAR10_Traditional" \
    "python src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --epochs 100 \
    --batch_size 128 \
    --noise_rate 0.0 \
    --lr 0.001"

# 2. ResNet18 CIFAR-10 Traditional (Baseline: 80.05%)
run_experiment "ResNet18_CIFAR10_Traditional" \
    "python src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type resnet \
    --epochs 100 \
    --batch_size 256 \
    --noise_rate 0.0 \
    --lr 0.001"

# 3. MLP MNIST Traditional (Baseline: 98.17%)
run_experiment "MLP_MNIST_Traditional" \
    "python src/experiments/run_experiment.py \
    --dataset mnist \
    --model_type mlp \
    --epochs 50 \
    --batch_size 128 \
    --noise_rate 0.0 \
    --lr 0.001"

# ============================================================================
# TABLE 2: ROBUST MODEL EXPERIMENTS (Noise-Robust Training)
# ============================================================================

echo "🔧 TABLE 2: ROBUST MODEL EXPERIMENTS"
echo "===================================="

# 4. CNN CIFAR-10 Robust CNN (GCE Loss)
run_experiment "CNN_CIFAR10_Robust_GCE" \
    "python src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type robust_cnn \
    --epochs 100 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --loss_type gce \
    --lr 0.001"

# 5. CNN CIFAR-10 Robust CNN (SCE Loss)
run_experiment "CNN_CIFAR10_Robust_SCE" \
    "python src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type robust_cnn \
    --epochs 100 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --loss_type sce \
    --lr 0.001"

# 6. ResNet18 CIFAR-10 Robust ResNet18 (GCE Loss)
run_experiment "ResNet18_CIFAR10_Robust_GCE" \
    "python src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type robust_resnet \
    --epochs 100 \
    --batch_size 256 \
    --noise_rate 0.1 \
    --loss_type gce \
    --lr 0.001"

# 7. ResNet18 CIFAR-10 Robust ResNet18 (SCE Loss)
run_experiment "ResNet18_CIFAR10_Robust_SCE" \
    "python src/experiments/run_experiment.py \
    --dataset cifar10 \
    --model_type robust_resnet \
    --epochs 100 \
    --batch_size 256 \
    --noise_rate 0.1 \
    --loss_type sce \
    --lr 0.001"

# 8. MLP MNIST Robust MLP (GCE Loss)
run_experiment "MLP_MNIST_Robust_GCE" \
    "python src/experiments/run_experiment.py \
    --dataset mnist \
    --model_type robust_mlp \
    --epochs 50 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --loss_type gce \
    --lr 0.001"

# 9. MLP MNIST Robust MLP (SCE Loss)
run_experiment "MLP_MNIST_Robust_SCE" \
    "python src/experiments/run_experiment.py \
    --dataset mnist \
    --model_type robust_mlp \
    --epochs 50 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --loss_type sce \
    --lr 0.001"

# ============================================================================
# TABLE 3: SEMI-SUPERVISED LEARNING EXPERIMENTS
# ============================================================================

echo "🎯 TABLE 3: SEMI-SUPERVISED LEARNING EXPERIMENTS"
echo "================================================"

# 10. CNN CIFAR-10 Consistency Regularization
run_experiment "CNN_CIFAR10_Consistency" \
    "python src/experiments/semi_supervised_experiments.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --labeled_ratio 0.1 \
    --strategy consistency \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.001"

# 11. CNN CIFAR-10 Pseudo-Labeling
run_experiment "CNN_CIFAR10_PseudoLabel" \
    "python src/experiments/semi_supervised_experiments.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --labeled_ratio 0.1 \
    --strategy pseudo_label \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.001 \
    --pseudo_threshold 0.95"

# 12. CNN CIFAR-10 Co-Training
run_experiment "CNN_CIFAR10_CoTraining" \
    "python src/experiments/semi_supervised_experiments.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --labeled_ratio 0.1 \
    --strategy co_training \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.001"

# 13. ResNet18 CIFAR-10 Consistency Regularization
run_experiment "ResNet18_CIFAR10_Consistency" \
    "python src/experiments/semi_supervised_experiments.py \
    --dataset cifar10 \
    --model_type resnet \
    --labeled_ratio 0.1 \
    --strategy consistency \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.001"

# 14. ResNet18 CIFAR-10 Pseudo-Labeling
run_experiment "ResNet18_CIFAR10_PseudoLabel" \
    "python src/experiments/semi_supervised_experiments.py \
    --dataset cifar10 \
    --model_type resnet \
    --labeled_ratio 0.1 \
    --strategy pseudo_label \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.001 \
    --pseudo_threshold 0.95"

# 15. ResNet18 CIFAR-10 Co-Training
run_experiment "ResNet18_CIFAR10_CoTraining" \
    "python src/experiments/semi_supervised_experiments.py \
    --dataset cifar10 \
    --model_type resnet \
    --labeled_ratio 0.1 \
    --strategy co_training \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.001"

# 16. MLP MNIST Consistency Regularization
run_experiment "MLP_MNIST_Consistency" \
    "python src/experiments/semi_supervised_experiments.py \
    --dataset mnist \
    --model_type mlp \
    --labeled_ratio 0.1 \
    --strategy consistency \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.001"

# 17. MLP MNIST Pseudo-Labeling
run_experiment "MLP_MNIST_PseudoLabel" \
    "python src/experiments/semi_supervised_experiments.py \
    --dataset mnist \
    --model_type mlp \
    --labeled_ratio 0.1 \
    --strategy pseudo_label \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.001 \
    --pseudo_threshold 0.95"

# 18. MLP MNIST Co-Training
run_experiment "MLP_MNIST_CoTraining" \
    "python src/experiments/semi_supervised_experiments.py \
    --dataset mnist \
    --model_type mlp \
    --labeled_ratio 0.1 \
    --strategy co_training \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.001"

# ============================================================================
# TABLE 4: UNIFIED WSL EXPERIMENTS (Combined Strategies)
# ============================================================================

echo "🚀 TABLE 4: UNIFIED WSL EXPERIMENTS"
echo "===================================="

# 19. CNN CIFAR-10 Combined WSL (All Strategies)
run_experiment "CNN_CIFAR10_CombinedWSL" \
    "python src/experiments/unified_wsl_experiment.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --strategies consistency pseudo_label co_training \
    --epochs 100 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --lr 0.0001"

# 20. ResNet18 CIFAR-10 Combined WSL (All Strategies)
run_experiment "ResNet18_CIFAR10_CombinedWSL" \
    "python src/experiments/unified_wsl_experiment.py \
    --dataset cifar10 \
    --model_type resnet \
    --strategies consistency pseudo_label co_training \
    --epochs 100 \
    --batch_size 256 \
    --noise_rate 0.1 \
    --lr 0.0001"

# 21. MLP MNIST Combined WSL (All Strategies)
run_experiment "MLP_MNIST_CombinedWSL" \
    "python src/experiments/unified_wsl_experiment.py \
    --dataset mnist \
    --model_type mlp \
    --strategies consistency pseudo_label co_training \
    --epochs 50 \
    --batch_size 128 \
    --noise_rate 0.1 \
    --lr 0.0001"

# ============================================================================
# TABLE 5: NOISE ROBUSTNESS ANALYSIS EXPERIMENTS
# ============================================================================

echo "🔬 TABLE 5: NOISE ROBUSTNESS ANALYSIS"
echo "====================================="

# 22. CIFAR-10 GCE Loss Noise Robustness (0%, 10%, 20% noise)
run_experiment "CIFAR10_NoiseRobustness_GCE" \
    "python src/experiments/noise_robustness_test.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --loss_type gce \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 50 \
    --batch_size 128"

# 23. CIFAR-10 SCE Loss Noise Robustness (0%, 10%, 20% noise)
run_experiment "CIFAR10_NoiseRobustness_SCE" \
    "python src/experiments/noise_robustness_test.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --loss_type sce \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 50 \
    --batch_size 128"

# 24. CIFAR-10 Forward Correction Noise Robustness (0%, 10%, 20% noise)
run_experiment "CIFAR10_NoiseRobustness_ForwardCorrection" \
    "python src/experiments/noise_robustness_test.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --loss_type forward_correction \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 50 \
    --batch_size 128"

# 25. MNIST GCE Loss Noise Robustness (0%, 10%, 20% noise)
run_experiment "MNIST_NoiseRobustness_GCE" \
    "python src/experiments/noise_robustness_test.py \
    --dataset mnist \
    --model_type mlp \
    --loss_type gce \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 30 \
    --batch_size 128"

# 26. MNIST SCE Loss Noise Robustness (0%, 10%, 20% noise)
run_experiment "MNIST_NoiseRobustness_SCE" \
    "python src/experiments/noise_robustness_test.py \
    --dataset mnist \
    --model_type mlp \
    --loss_type sce \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 30 \
    --batch_size 128"

# 27. MNIST Forward Correction Noise Robustness (0%, 10%, 20% noise)
run_experiment "MNIST_NoiseRobustness_ForwardCorrection" \
    "python src/experiments/noise_robustness_test.py \
    --dataset mnist \
    --model_type mlp \
    --loss_type forward_correction \
    --noise_levels 0.0 0.1 0.2 \
    --epochs 30 \
    --batch_size 128"

# ============================================================================
# TABLE 6: ANALYSIS AND EVALUATION EXPERIMENTS
# ============================================================================

echo "📈 TABLE 6: ANALYSIS AND EVALUATION"
echo "===================================="

# 28. Generate Comprehensive Performance Report
run_experiment "Generate_Performance_Report" \
    "python src/experiments/generate_performance_report.py \
    --datasets cifar10 mnist \
    --model_types simple_cnn robust_cnn resnet robust_resnet mlp robust_mlp \
    --strategies traditional consistency pseudo_label co_training combined \
    --output_file performance_comparison_report.json"

# 29. Generate Training Curves Visualization
run_experiment "Generate_Training_Curves" \
    "python src/visualization/generate_training_curves.py \
    --datasets cifar10 mnist \
    --strategies traditional consistency pseudo_label co_training combined \
    --epochs 100 \
    --output_dir ./training_curves"

# 30. Generate Confusion Matrices
run_experiment "Generate_CIFAR10_Confusion_Matrix" \
    "python src/experiments/generate_confusion_matrices.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --strategy combined \
    --output_file cifar10_confusion_matrix.png"

run_experiment "Generate_MNIST_Confusion_Matrix" \
    "python src/experiments/generate_confusion_matrices.py \
    --dataset mnist \
    --model_type robust_mlp \
    --strategy combined \
    --output_file mnist_confusion_matrix.png"

# 31. Feature Engineering Analysis
run_experiment "Feature_Engineering_Analysis" \
    "python src/experiments/feature_engineering_analysis.py \
    --strategies consistency pseudo_label co_training combined \
    --datasets cifar10 mnist \
    --output_file feature_engineering_results.json"

# 32. Generate Feature Engineering Plots
run_experiment "Generate_Feature_Engineering_Plots" \
    "python src/visualization/feature_engineering_plots.py \
    --input_file feature_engineering_results.json \
    --output_dir ./feature_engineering_plots"

# 33. CIFAR-10 Data Augmentation Analysis
run_experiment "CIFAR10_Data_Augmentation_Analysis" \
    "python src/experiments/data_augmentation_analysis.py \
    --dataset cifar10 \
    --model_type simple_cnn \
    --augmentations random_rotation horizontal_flip random_crop color_jitter \
    --epochs 50 \
    --batch_size 128"

# 34. MNIST Data Augmentation Analysis
run_experiment "MNIST_Data_Augmentation_Analysis" \
    "python src/experiments/data_augmentation_analysis.py \
    --dataset mnist \
    --model_type mlp \
    --augmentations random_rotation gaussian_noise \
    --epochs 30 \
    --batch_size 128"

# 35. Generate Augmentation Comparison Plots
run_experiment "Generate_Augmentation_Plots" \
    "python src/visualization/augmentation_plots.py \
    --output_dir ./augmentation_plots"

# 36. Hardware Configuration Testing
run_experiment "Hardware_Configuration_Test" \
    "python src/experiments/hardware_configuration_test.py \
    --cpu_test \
    --gpu_test \
    --memory_test \
    --storage_test \
    --output_file hardware_test_results.json"

# 37. Memory Usage Analysis
run_experiment "Memory_Usage_Analysis" \
    "python src/experiments/memory_analysis.py \
    --datasets cifar10 mnist \
    --model_types simple_cnn robust_cnn resnet robust_resnet mlp robust_mlp \
    --batch_sizes 32 64 128 256"

# 38. Training Time Analysis
run_experiment "Training_Time_Analysis" \
    "python src/experiments/training_time_analysis.py \
    --datasets cifar10 mnist \
    --model_types simple_cnn robust_cnn resnet robust_resnet mlp robust_mlp \
    --epochs 20"

# 39. Dataset Quality Analysis
run_experiment "Dataset_Quality_Analysis" \
    "python src/experiments/dataset_quality_analysis.py \
    --datasets cifar10 mnist \
    --metrics completeness relevance consistency diversity \
    --output_file dataset_quality_results.json"

# 40. Model Architecture Analysis
run_experiment "Model_Architecture_Analysis" \
    "python src/experiments/model_architecture_analysis.py \
    --models simple_cnn robust_cnn resnet robust_resnet mlp robust_mlp \
    --datasets cifar10 mnist \
    --output_file model_architecture_results.json"

# 41. Generate Architecture Comparison Plots
run_experiment "Generate_Architecture_Plots" \
    "python src/visualization/model_architecture_plots.py \
    --input_file model_architecture_results.json \
    --output_dir ./architecture_plots"

# ============================================================================
# TABLE 7: COMPREHENSIVE TESTING SUITE
# ============================================================================

echo "🧪 TABLE 7: COMPREHENSIVE TESTING SUITE"
echo "========================================"

# 42. Comprehensive Testing Suite
run_experiment "Comprehensive_Testing_Suite" \
    "python src/tests/run_comprehensive_tests.py \
    --test_modules data_preprocessing strategy_selection model_training evaluation \
    --test_types unit integration system performance \
    --output_file comprehensive_test_results.json"

# 43. Generate Test Results Plots
run_experiment "Generate_Test_Results_Plots" \
    "python src/visualization/test_results_plots.py \
    --input_file comprehensive_test_results.json \
    --output_dir ./test_results_plots"

# ============================================================================
# TABLE 8: FINAL REPORT GENERATION
# ============================================================================

echo "📊 TABLE 8: FINAL REPORT GENERATION"
echo "==================================="

# 44. Generate Comprehensive Final Report
run_experiment "Generate_Comprehensive_Final_Report" \
    "python src/experiments/generate_comprehensive_report.py \
    --include_performance \
    --include_feature_engineering \
    --include_data_augmentation \
    --include_hardware_analysis \
    --include_testing_results \
    --output_file comprehensive_final_report.md"

# 45. Generate Executive Summary
run_experiment "Generate_Executive_Summary" \
    "python src/experiments/generate_executive_summary.py \
    --input_file comprehensive_final_report.md \
    --output_file executive_summary.md"

# 46. Generate Summary Report
run_experiment "Generate_Summary_Report" \
    "python src/experiments/generate_summary_report.py \
    --output_file final_summary_report.md"

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo "🎉 COMPLETE EXPERIMENT SUITE FINISHED!"
echo "======================================"
echo ""
echo "📊 Total Experiments Completed: 46"
echo "⏱️  Total Execution Time: ~48-72 hours"
echo "📁 Results saved in: experiment_results/"
echo "📋 Logs saved in: logs/"
echo ""
echo "📈 Key Results Generated:"
echo "  - Baseline performance comparisons"
echo "  - Robust model evaluations"
echo "  - Semi-supervised learning results"
echo "  - Unified WSL framework results"
echo "  - Noise robustness analysis"
echo "  - Comprehensive performance reports"
echo "  - Feature engineering analysis"
echo "  - Data augmentation impact"
echo "  - Hardware configuration testing"
echo "  - Final comprehensive reports"
echo ""
echo "🚀 Your WSL Framework experiments are complete!"
echo "📝 Use the generated reports for your paper and presentation."
echo "" 