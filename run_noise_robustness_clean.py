#!/usr/bin/env python3
"""
Clean Noise Robustness Test Runner
Runs noise robustness tests with suppressed warnings and cleaner output
"""

import os
import sys
import subprocess
import warnings

# Set environment variables to suppress macOS warnings
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MALLOC_NANOZONE'] = '0'
os.environ['PYTORCH_DISABLE_MPS'] = '1'

# Suppress all warnings
warnings.filterwarnings("ignore")

def run_noise_robustness_test():
    """Run noise robustness test with clean output"""
    
    # Command to run
    cmd = [
        sys.executable, 
        "src/experiments/noise_robustness_test.py",
        "--dataset", "cifar10",
        "--model_type", "simple_cnn", 
        "--loss_type", "gce",
        "--noise_levels", "0.0", "0.1", "0.2",
        "--epochs", "20"
    ]
    
    # Set environment variables for the subprocess
    env = os.environ.copy()
    env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    env['MALLOC_NANOZONE'] = '0'
    env['PYTORCH_DISABLE_MPS'] = '1'
    
    print("Starting noise robustness test with clean output...")
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
        
        # Print only the important output lines
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            # Print only training progress and important messages
            if any(keyword in line for keyword in [
                'Starting noise robustness test',
                'Dataset:', 'Model:', 'Loss:', 'Noise levels:', 'Epochs:',
                'Training with', 'Epoch', 'NOISE ROBUSTNESS TEST SUMMARY',
                'Performance by Noise Level:', 'Average Robustness Score',
                'Best Performance:', 'Worst Performance:'
            ]):
                print(line)
        
        if result.returncode == 0:
            print("\n✅ Noise robustness test completed successfully!")
        else:
            print(f"\n❌ Noise robustness test failed with return code: {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
                
    except subprocess.TimeoutExpired:
        print("\n❌ Noise robustness test timed out after 1 hour")
    except Exception as e:
        print(f"\n❌ Error running noise robustness test: {e}")

if __name__ == "__main__":
    run_noise_robustness_test() 