import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
import torch.nn as nn
from itertools import combinations

class ModelSelection:
    """Model selection with various criteria"""
    
    @staticmethod
    def cross_validation(
        model_fn: Callable[[], nn.Module],
        train_data: np.ndarray,
        train_labels: np.ndarray,
        n_splits: int = 5
    ) -> float:
        """Perform k-fold cross-validation"""
        kf = KFold(n_splits=n_splits, shuffle=True)
        scores = []
        
        for train_idx, val_idx in kf.split(train_data):
            X_train, X_val = train_data[train_idx], train_data[val_idx]
            y_train, y_val = train_labels[train_idx], train_labels[val_idx]
            
            # Create and train model
            model = model_fn()
            model.train()
            
            # Convert data to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            
            # Train model
            optimizer = torch.optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()
            
            for _ in range(5):  # Quick training for validation
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                _, predicted = torch.max(val_outputs, 1)
                score = accuracy_score(y_val, predicted.numpy())
                scores.append(score)
        
        return np.mean(scores)
    
    @staticmethod
    def model_complexity(
        model: torch.nn.Module,
        input_shape: Tuple[int, ...]
    ) -> Dict[str, float]:
        """
        Calculate model complexity metrics
        Args:
            model: PyTorch model
            input_shape: Shape of input tensor
        Returns:
            Dictionary of complexity metrics
        """
        total_params = sum(p.numel() for p in model.parameters())
        
        # Calculate FLOPs
        input_tensor = torch.randn(1, *input_shape)
        flops = 0
        
        def count_flops(module, input, output):
            nonlocal flops
            if isinstance(module, nn.Conv2d):
                # Get output dimensions
                if isinstance(output, tuple):
                    output = output[0]
                out_h = output.size(2)
                out_w = output.size(3)
                
                flops += (module.in_channels * module.out_channels * 
                         module.kernel_size[0] * module.kernel_size[1] * 
                         out_h * out_w)
            elif isinstance(module, nn.Linear):
                flops += module.in_features * module.out_features
        
        # Register hook
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(count_flops))
        
        # Forward pass
        model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return {
            'total_params': total_params,
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'flops': flops
        }
    
    @staticmethod
    def ensemble_selection(
        models: List[torch.nn.Module],
        val_data: Union[torch.Tensor, np.ndarray],
        val_labels: Union[torch.Tensor, np.ndarray],
        metric: str = 'accuracy'
    ) -> Tuple[List[int], float]:
        """
        Select best ensemble of models
        Args:
            models: List of trained models
            val_data: Validation data (PyTorch tensor or numpy array)
            val_labels: Validation labels (PyTorch tensor or numpy array)
            metric: Evaluation metric
        Returns:
            Tuple of (selected model indices, ensemble score)
        """
        val_data_tensor = torch.FloatTensor(val_data)
        val_labels_tensor = torch.LongTensor(val_labels)
        
        # Get predictions from all models
        all_preds = []
        for model in models:
            model.eval()
            with torch.no_grad():
                outputs = model(val_data_tensor)
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds.numpy())
        
        # Try different combinations
        best_score = 0
        best_ensemble = []
        
        for i in range(1, len(models) + 1):
            for combo in combinations(range(len(models)), i):
                ensemble_preds = np.mean([all_preds[j] for j in combo], axis=0)
                ensemble_preds = np.round(ensemble_preds).astype(int)
                score = accuracy_score(val_labels, ensemble_preds)
                
                if score > best_score:
                    best_score = score
                    best_ensemble = [models[j] for j in combo]
        
        return best_ensemble, best_score
    
    @staticmethod
    def hyperparameter_tuning(
        model_fn,
        param_grid: Dict[str, List[Any]],
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: torch.Tensor,
        val_labels: torch.Tensor,
        metric: str = 'accuracy'
    ) -> Tuple[Dict[str, Any], float]:
        """
        Perform hyperparameter tuning
        Args:
            model_fn: Function that returns a model instance
            param_grid: Dictionary of parameter names and possible values
            train_data: Training data
            train_labels: Training labels
            val_data: Validation data
            val_labels: Validation labels
            metric: Evaluation metric
        Returns:
            Tuple of (best parameters, best score)
        """
        best_score = 0
        best_params = None
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        from itertools import product
        for params in product(*param_values):
            # Create parameter dictionary
            param_dict = dict(zip(param_names, params))
            
            # Create and train model
            model = model_fn(**param_dict)
            model.train()
            # ... training code ...
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                val_preds = model.predict(val_data)
            
            # Calculate score
            if metric == 'accuracy':
                score = accuracy_score(val_labels, val_preds)
            elif metric == 'f1':
                score = f1_score(val_labels, val_preds, average='weighted')
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            # Update best parameters
            if score > best_score:
                best_score = score
                best_params = param_dict
        
        return best_params, best_score 