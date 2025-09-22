import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import warnings

@dataclass
class OptimizationConfig:
    """Configuration for performance optimization"""
    use_mixed_precision: bool = True
    use_gradient_accumulation: bool = True
    accumulation_steps: int = 4
    use_model_pruning: bool = False
    pruning_ratio: float = 0.3
    use_quantization: bool = False
    quantization_type: str = 'dynamic'
    use_caching: bool = True
    cache_size: int = 1000
    use_early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001

class PerformanceOptimizer:
    """Advanced performance optimization for unified WSL framework"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache = {}
        self.performance_history = []
        self.best_performance = float('-inf')
        self.patience_counter = 0
        
        # Initialize mixed precision scaler if available
        if self.config.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def optimize_training_step(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        labeled_data: torch.Tensor,
        labeled_targets: torch.Tensor,
        unlabeled_data: Optional[torch.Tensor] = None,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """Optimized training step with multiple optimization techniques"""
        
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Mixed precision training
        if self.config.use_mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                loss = self._compute_loss(
                    model, labeled_data, labeled_targets, 
                    unlabeled_data, criterion
                )
            
            # Scale loss and backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if self.config.use_gradient_accumulation:
                if (self.accumulation_step + 1) % self.config.accumulation_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                self.scaler.step(optimizer)
                self.scaler.update()
        else:
            # Standard training
            loss = self._compute_loss(
                model, labeled_data, labeled_targets, 
                unlabeled_data, criterion
            )
            loss.backward()
            
            # Gradient accumulation
            if self.config.use_gradient_accumulation:
                if (self.accumulation_step + 1) % self.config.accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.step()
        
        # Update accumulation step counter
        if not hasattr(self, 'accumulation_step'):
            self.accumulation_step = 0
        self.accumulation_step += 1
        
        # Performance monitoring
        end_time = time.time()
        memory_after = self._get_memory_usage()
        
        performance_metrics = {
            'loss': loss.item(),
            'training_time': end_time - start_time,
            'memory_usage': memory_after - memory_before,
            'gpu_memory': self._get_gpu_memory_usage() if torch.cuda.is_available() else 0
        }
        
        self.performance_history.append(performance_metrics)
        
        return performance_metrics
    
    def _compute_loss(
        self,
        model: nn.Module,
        labeled_data: torch.Tensor,
        labeled_targets: torch.Tensor,
        unlabeled_data: Optional[torch.Tensor] = None,
        criterion: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """Compute loss with caching for repeated computations"""
        
        # Create cache key
        cache_key = self._create_cache_key(labeled_data, labeled_targets, unlabeled_data)
        
        # Check cache
        if self.config.use_caching and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Compute loss
        if hasattr(model, 'forward') and callable(getattr(model, 'forward')):
            # For unified WSL models
            if unlabeled_data is not None:
                loss = model(labeled_data, labeled_targets, unlabeled_data)
            else:
                loss = model(labeled_data, labeled_targets)
        else:
            # For standard models
            outputs = model(labeled_data)
            if criterion is None:
                criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labeled_targets)
        
        # Cache result
        if self.config.use_caching:
            self._update_cache(cache_key, loss)
        
        return loss
    
    def _create_cache_key(
        self,
        labeled_data: torch.Tensor,
        labeled_targets: torch.Tensor,
        unlabeled_data: Optional[torch.Tensor] = None
    ) -> str:
        """Create cache key for data"""
        key_parts = [
            str(labeled_data.shape),
            str(labeled_targets.shape),
            str(unlabeled_data.shape) if unlabeled_data is not None else 'None'
        ]
        return hash(tuple(key_parts))
    
    def _update_cache(self, key: str, value: torch.Tensor):
        """Update cache with size limit"""
        if len(self.cache) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def apply_model_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply model-level optimizations"""
        
        # Model pruning
        if self.config.use_model_pruning:
            model = self._apply_pruning(model)
        
        # Model quantization
        if self.config.use_quantization:
            model = self._apply_quantization(model)
        
        return model
    
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply model pruning"""
        total_params = sum(p.numel() for p in model.parameters())
        target_params = int(total_params * (1 - self.config.pruning_ratio))
        
        # Simple magnitude-based pruning
        parameters = []
        for param in model.parameters():
            if param.dim() > 1:  # Only prune weight matrices
                parameters.append(param.view(-1))
        
        if parameters:
            all_weights = torch.cat(parameters)
            threshold = torch.kthvalue(all_weights.abs(), target_params)[0]
            
            for param in model.parameters():
                if param.dim() > 1:
                    mask = param.abs() > threshold
                    param.data *= mask.float()
        
        return model
    
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply model quantization"""
        if self.config.quantization_type == 'dynamic':
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
        elif self.config.quantization_type == 'static':
            # Static quantization requires calibration
            model.eval()
            model = torch.quantization.quantize_static(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
        
        return model
    
    def optimize_memory_usage(self):
        """Optimize memory usage"""
        # Clear cache if too large
        if len(self.cache) > self.config.cache_size * 2:
            self.cache.clear()
        
        # Clear performance history if too long
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.performance_history:
            return {}
        
        # Calculate statistics
        losses = [p['loss'] for p in self.performance_history]
        times = [p['training_time'] for p in self.performance_history]
        memory_usage = [p['memory_usage'] for p in self.performance_history]
        
        report = {
            'total_steps': len(self.performance_history),
            'average_loss': np.mean(losses),
            'loss_std': np.std(losses),
            'min_loss': np.min(losses),
            'max_loss': np.max(losses),
            'average_training_time': np.mean(times),
            'total_training_time': np.sum(times),
            'average_memory_usage': np.mean(memory_usage),
            'peak_memory_usage': np.max(memory_usage),
            'cache_size': len(self.cache),
            'optimization_config': self.config
        }
        
        # Add GPU memory statistics if available
        if torch.cuda.is_available():
            gpu_memory = [p['gpu_memory'] for p in self.performance_history]
            report.update({
                'average_gpu_memory': np.mean(gpu_memory),
                'peak_gpu_memory': np.max(gpu_memory)
            })
        
        return report
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
        return 0.0
    
    def check_early_stopping(self, current_performance: float) -> bool:
        """Check if early stopping should be triggered"""
        if not self.config.use_early_stopping:
            return False
        
        if current_performance > self.best_performance + self.config.min_delta:
            self.best_performance = current_performance
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.config.patience
    
    def reset_early_stopping(self):
        """Reset early stopping counters"""
        self.best_performance = float('-inf')
        self.patience_counter = 0

class AdaptiveBatchSizeOptimizer:
    """Optimize batch size based on memory and performance"""
    
    def __init__(self, initial_batch_size: int = 32, max_batch_size: int = 512):
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        self.performance_history = []
    
    def optimize_batch_size(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        target_memory_usage: float = 0.8
    ) -> int:
        """Find optimal batch size based on memory constraints"""
        
        device = next(model.parameters()).device
        
        # Test different batch sizes
        batch_sizes = [16, 32, 64, 128, 256, 512]
        valid_batch_sizes = []
        
        for batch_size in batch_sizes:
            if batch_size > self.max_batch_size:
                continue
            
            try:
                # Create test batch
                test_data = torch.randn(batch_size, *data_loader.dataset[0][0].shape).to(device)
                
                # Measure memory usage
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                memory_before = self._get_memory_usage()
                
                with torch.no_grad():
                    _ = model(test_data)
                
                memory_after = self._get_memory_usage()
                memory_usage = memory_after - memory_before
                
                # Check if memory usage is acceptable
                if memory_usage < target_memory_usage * self._get_total_memory():
                    valid_batch_sizes.append((batch_size, memory_usage))
                
                del test_data
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    continue
                else:
                    raise e
        
        if not valid_batch_sizes:
            return self.initial_batch_size
        
        # Choose batch size with highest memory efficiency
        optimal_batch_size = max(valid_batch_sizes, key=lambda x: x[0] / x[1])[0]
        self.current_batch_size = optimal_batch_size
        
        return optimal_batch_size
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
    
    def _get_total_memory(self) -> float:
        """Get total available memory"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
        else:
            return psutil.virtual_memory().total / 1024 / 1024  # MB

class LearningRateOptimizer:
    """Optimize learning rate using various strategies"""
    
    def __init__(self, initial_lr: float = 0.001):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.lr_history = []
    
    def find_optimal_lr(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer_class: type = torch.optim.Adam,
        lr_range: Tuple[float, float] = (1e-6, 1e-1),
        num_steps: int = 100
    ) -> float:
        """Find optimal learning rate using learning rate finder"""
        
        # Create temporary optimizer
        optimizer = optimizer_class(model.parameters(), lr=lr_range[0])
        
        # Learning rate finder
        lr_finder = torch.optim.lr_finder.LRFinder(
            model, optimizer, criterion, device=next(model.parameters()).device
        )
        
        # Run learning rate finder
        lr_finder.range_test(train_loader, end_lr=lr_range[1], num_iter=num_steps)
        
        # Get optimal learning rate
        optimal_lr = lr_finder.suggestion()
        lr_finder.reset()
        
        self.current_lr = optimal_lr
        self.lr_history.append(optimal_lr)
        
        return optimal_lr
    
    def create_adaptive_scheduler(
        self,
        optimizer: Optimizer,
        scheduler_type: str = 'cosine',
        **kwargs
    ) -> _LRScheduler:
        """Create adaptive learning rate scheduler"""
        
        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=kwargs.get('T_max', 100), eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=kwargs.get('factor', 0.5), patience=kwargs.get('patience', 10)
            )
        elif scheduler_type == 'one_cycle':
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.current_lr, epochs=kwargs.get('epochs', 100), steps_per_epoch=kwargs.get('steps_per_epoch', 100)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}") 