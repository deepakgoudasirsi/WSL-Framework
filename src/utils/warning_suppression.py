#!/usr/bin/env python3
"""
Warning Suppression Utilities
Comprehensive solution for suppressing macOS warnings in PyTorch experiments
"""

import os
import sys
import contextlib
from typing import Optional

def setup_macos_warning_suppression():
    """
    Set up comprehensive macOS warning suppression
    This should be called at the very beginning of any script
    """
    # Set environment variables for current process and subprocesses
    os.environ['MALLOC_NANOZONE'] = '0'
    os.environ['MALLOC_STACK_LOGGING'] = '0'
    os.environ['MALLOC_PROTECT_BEFORE'] = '0'
    os.environ['MALLOC_FILL_SPACE'] = '0'
    os.environ['MALLOC_LOGGING'] = '0'
    os.environ['MALLOC_STRICT_SIZE'] = '0'
    
    # Additional environment variables for subprocesses
    os.environ['PYTORCH_DISABLE_WARNINGS'] = '1'
    os.environ['PYTHONWARNINGS'] = 'ignore'

class FilteredStderr:
    """Filter stderr to suppress specific warnings"""
    
    def __init__(self):
        self.original_stderr = sys.stderr
        self.filtered_lines = []
    
    def write(self, text):
        # Filter out MallocStackLogging warnings and other macOS warnings
        if any(warning in text.lower() for warning in [
            'mallocstacklogging', 'malloc', 'nanozone', 'protect_before', 'fill_space'
        ]):
            self.filtered_lines.append(text)
        else:
            self.original_stderr.write(text)
    
    def flush(self):
        self.original_stderr.flush()

@contextlib.contextmanager
def suppress_macos_warnings():
    """
    Context manager to suppress macOS warnings during execution
    Usage:
        with suppress_macos_warnings():
            # Your code here
    """
    # Setup environment variables
    setup_macos_warning_suppression()
    
    # Store original stderr
    original_stderr = sys.stderr
    
    try:
        # Apply filtered stderr
        sys.stderr = FilteredStderr()
        yield
    finally:
        # Restore original stderr
        sys.stderr = original_stderr

def apply_warning_suppression():
    """
    Apply warning suppression globally
    Call this function at the start of any script
    """
    setup_macos_warning_suppression()
    sys.stderr = FilteredStderr()

# Apply suppression when module is imported
if __name__ != '__main__':
    apply_warning_suppression() 