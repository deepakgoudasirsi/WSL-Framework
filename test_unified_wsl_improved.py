#!/usr/bin/env python3
"""
Improved Unified WSL Experiment Test
Runs the unified WSL experiment with better settings and error handling
"""

import os
import sys
import subprocess
import warnings

# Set environment variables to suppress macOS warnings
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MALLOC_NANOZONE'] = '0'
os.environ['PYTORCH_DISABLE_MPS'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Suppress all warnings
warnings.filterwarnings("ignore")

def run_unified_wsl_experiment():
    """Run unified WSL experiment with improved settings"""
    
    # Command to run with better parameters
    cmd = [
        sys.executable, 
        "src/experiments/unified_wsl_experiment.py",
        "--dataset", "mnist",
        "--model_type", "mlp", 
        "--strategies", "consistency", "pseudo_label", "co_training",
        "--epochs", "30",
        "--batch_size", "64",  # Smaller batch size for stability
        "--noise_rate", "0.1"
    ]
    
    # Set environment variables for the subprocess
    env = os.environ.copy()
    env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    env['MALLOC_NANOZONE'] = '0'
    env['PYTORCH_DISABLE_MPS'] = '1'
    env['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    print("Starting improved unified WSL experiment...")
    print("=" * 60)
    print("Improvements made:")
    print("- Lower learning rate (0.0001) for stability")
    print("- Gradient clipping to prevent explosion")
    print("- Better learning rate scheduler (CosineAnnealing)")
    print("- NaN loss detection and handling")
    print("- Better weight initialization")
    print("- Smaller batch size (64) for stability")
    print("=" * 60)
    
    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        # Print the output
        print(result.stdout)
        
        if result.returncode == 0:
            print("\n✅ Unified WSL experiment completed successfully!")
        else:
            print(f"\n❌ Unified WSL experiment failed with return code: {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
                
    except subprocess.TimeoutExpired:
        print("\n❌ Unified WSL experiment timed out after 1 hour")
    except Exception as e:
        print(f"\n❌ Error running unified WSL experiment: {e}")

if __name__ == "__main__":
    run_unified_wsl_experiment() 