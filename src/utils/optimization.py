import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import time
from src.models.unified_wsl import UnifiedWSLModel

class GradientAccumulation:
    """Gradient accumulation for training with large batch sizes"""
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        accumulation_steps: int = 1
    ):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def step(self, loss: torch.Tensor):
        """Accumulate gradients and update weights"""
        # Normalize loss by accumulation steps
        loss = loss / self.accumulation_steps
        loss.backward()
        
        self.current_step += 1
        if self.current_step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

class MixedPrecisionTraining:
    """Mixed precision training with automatic mixed precision (AMP)"""
    
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.device = next(model.parameters()).device
        self.use_amp = torch.cuda.is_available()
        
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
    
    def train_step(self, labeled_data, labeled_targets, unlabeled_data, criterion):
        """Perform a single training step with mixed precision"""
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                # For UnifiedWSLModel, we need to pass all three arguments
                if isinstance(self.model, UnifiedWSLModel):
                    loss = self.model(labeled_data, labeled_targets, unlabeled_data)
                else:
                    # For regular models, use standard forward pass
                    outputs = self.model(labeled_data)
                    loss = criterion(outputs, labeled_targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular training without mixed precision
            if isinstance(self.model, UnifiedWSLModel):
                loss = self.model(labeled_data, labeled_targets, unlabeled_data)
            else:
                outputs = self.model(labeled_data)
                loss = criterion(outputs, labeled_targets)
            loss.backward()
            self.optimizer.step()
        
        return loss

class LearningRateScheduler:
    """Advanced learning rate scheduling"""
    def __init__(
        self,
        optimizer: Optimizer,
        scheduler_type: str = 'cosine',
        **kwargs
    ):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                **kwargs
            )
        elif scheduler_type == 'one_cycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                **kwargs
            )
        elif scheduler_type == 'cyclic':
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def step(self, metrics: Optional[Dict[str, float]] = None):
        """Step the learning rate scheduler"""
        if self.scheduler_type == 'cosine':
            self.scheduler.step()
        elif self.scheduler_type == 'one_cycle':
            self.scheduler.step()
        elif self.scheduler_type == 'cyclic':
            self.scheduler.step()

class ModelPruning:
    """Model pruning for efficiency"""
    def __init__(
        self,
        model: nn.Module,
        pruning_type: str = 'l1',
        pruning_ratio: float = 0.3
    ):
        self.model = model
        self.pruning_type = pruning_type
        self.pruning_ratio = pruning_ratio
    
    def prune(self):
        """Prune the model"""
        if self.pruning_type == 'l1':
            self._l1_pruning()
        elif self.pruning_type == 'random':
            self._random_pruning()
        else:
            raise ValueError(f"Unsupported pruning type: {self.pruning_type}")
    
    def _l1_pruning(self):
        """L1 norm-based pruning"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Calculate L1 norms
                weights = module.weight.data
                norms = torch.norm(weights, p=1, dim=1)
                
                # Get threshold
                threshold = torch.quantile(norms, self.pruning_ratio)
                
                # Create mask
                mask = norms > threshold
                
                # Apply mask
                module.weight.data *= mask.unsqueeze(1)
    
    def _random_pruning(self):
        """Random pruning"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Create random mask
                mask = torch.rand_like(module.weight.data) > self.pruning_ratio
                
                # Apply mask
                module.weight.data *= mask

class ModelQuantization:
    """Model quantization utilities"""
    
    def __init__(
        self,
        model: nn.Module,
        quantization_type: str = 'dynamic'
    ):
        self.model = model
        self.quantization_type = quantization_type
    
    def quantize(self):
        """Quantize the model"""
        if self.quantization_type == 'dynamic':
            self._dynamic_quantization()
        else:
            raise ValueError(f"Unsupported quantization type: {self.quantization_type}")
    
    def _dynamic_quantization(self):
        """Apply dynamic quantization"""
        try:
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear},  # Quantize only linear layers
                dtype=torch.qint8
            )
        except RuntimeError as e:
            if "NoQEngine" in str(e):
                print("Warning: Quantization engine not available. Skipping quantization.")
            else:
                raise e

class PerformanceMonitor:
    """Monitor model performance metrics"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
    
    def measure_performance(
        self,
        data: torch.Tensor,
        batch_size: int = 32,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """Measure model performance metrics"""
        self.model.eval()
        
        # Use CPU timing if CUDA is not available
        if not torch.cuda.is_available():
            return self._measure_cpu_performance(data, batch_size, num_iterations)
        
        # CUDA performance measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_event.record()
                _ = self.model(data)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))
        
        return {
            'mean_latency': np.mean(times),
            'std_latency': np.std(times),
            'min_latency': np.min(times),
            'max_latency': np.max(times)
        }
    
    def _measure_cpu_performance(
        self,
        data: torch.Tensor,
        batch_size: int,
        num_iterations: int
    ) -> Dict[str, float]:
        """Measure performance on CPU"""
        self.model.eval()
        times = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                _ = self.model(data)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'mean_latency': np.mean(times),
            'std_latency': np.std(times),
            'min_latency': np.min(times),
            'max_latency': np.max(times)
        } 