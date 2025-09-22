#!/bin/bash
# Script to run experiments with macOS warning suppression

# Set environment variables to suppress macOS warnings
export MALLOC_NANOZONE=0
export MALLOC_STACK_LOGGING=0
export MALLOC_PROTECT_BEFORE=0
export MALLOC_FILL_SPACE=0
export MALLOC_LOGGING=0
export MALLOC_STRICT_SIZE=0
export PYTORCH_DISABLE_WARNINGS=1
export PYTHONWARNINGS=ignore

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Function to run experiment with warning suppression
run_experiment() {
    echo "Running experiment with warning suppression..."
    echo "Command: $@"
    echo "----------------------------------------"
    
    # Run the experiment
    python "$@"
    
    echo "----------------------------------------"
    echo "Experiment completed!"
}

# Check if arguments are provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <experiment_script> [arguments...]"
    echo "Example: $0 src/experiments/semi_supervised_experiments.py --dataset mnist --strategy pseudo_label"
    exit 1
fi

# Run the experiment
run_experiment "$@" 