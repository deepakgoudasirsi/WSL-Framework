import numpy as np
from typing import List, Callable, Optional, Union
from dataclasses import dataclass

@dataclass
class LabelingFunction:
    """Base class for labeling functions"""
    name: str
    function: Callable
    label: int

def create_keyword_lf(keywords: List[str], label: int, name: Optional[str] = None) -> LabelingFunction:
    """Create a keyword-based labeling function"""
    def keyword_fn(text: str) -> int:
        return label if any(kw.lower() in text.lower() for kw in keywords) else -1
    
    return LabelingFunction(
        name=name or f"keyword_{label}",
        function=keyword_fn,
        label=label
    )

def create_regex_lf(pattern: str, label: int, name: Optional[str] = None) -> LabelingFunction:
    """Create a regex-based labeling function"""
    import re
    def regex_fn(text: str) -> int:
        return label if re.search(pattern, text) else -1
    
    return LabelingFunction(
        name=name or f"regex_{label}",
        function=regex_fn,
        label=label
    )

def create_heuristic_lf(heuristic_fn: Callable, label: int, name: Optional[str] = None) -> LabelingFunction:
    """Create a heuristic-based labeling function"""
    def wrapped_fn(x: Union[str, np.ndarray]) -> int:
        return label if heuristic_fn(x) else -1
    
    return LabelingFunction(
        name=name or f"heuristic_{label}",
        function=wrapped_fn,
        label=label
    )

class DataProgrammingDataset:
    """Dataset for data programming with labeling functions"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.lfs = {}  # Dictionary to store labeling functions
        self.lf_outputs = None  # Cache for LF outputs
    
    def add_lf(self, name: str, lf: Callable, description: str = ""):
        """Add a labeling function to the dataset"""
        self.lfs[name] = {
            'function': lf,
            'description': description
        }
        # Clear cached outputs when adding new LF
        self.lf_outputs = None
    
    def get_lf_outputs(self) -> np.ndarray:
        """Get outputs from all labeling functions"""
        if self.lf_outputs is None:
            n_samples = len(self.X)
            n_lfs = len(self.lfs)
            self.lf_outputs = np.zeros((n_samples, n_lfs))
            
            for i, (name, lf_info) in enumerate(self.lfs.items()):
                print(f"Applying labeling function: {name}")
                for j in range(n_samples):
                    try:
                        self.lf_outputs[j, i] = lf_info['function'](self.X[j])
                    except Exception as e:
                        print(f"Error applying LF {name} to sample {j}: {e}")
                        self.lf_outputs[j, i] = -1  # Use -1 for abstain
        
        return self.lf_outputs
    
    def get_labels(self, aggregation_method: str = 'weighted_vote') -> np.ndarray:
        """Get labels from labeling functions using specified aggregation method"""
        if not self.lfs:
            raise ValueError("No labeling functions available")
        
        lf_outputs = self.get_lf_outputs()
        n_samples = len(self.X)
        labels = np.zeros(n_samples)
        
        if aggregation_method == 'majority_vote':
            # Simple majority voting
            for i in range(n_samples):
                valid_outputs = lf_outputs[i][lf_outputs[i] != -1]
                if len(valid_outputs) > 0:
                    labels[i] = np.bincount(valid_outputs.astype(int)).argmax()
                else:
                    labels[i] = -1  # Abstain if no valid outputs
        
        elif aggregation_method == 'weighted_vote':
            # Weighted voting based on LF accuracy
            lf_weights = self._compute_lf_weights()
            for i in range(n_samples):
                valid_mask = lf_outputs[i] != -1
                if np.any(valid_mask):
                    valid_outputs = lf_outputs[i][valid_mask]
                    valid_weights = lf_weights[valid_mask]
                    weighted_sum = np.zeros(10)  # Assuming 10 classes
                    for j, output in enumerate(valid_outputs):
                        weighted_sum[int(output)] += valid_weights[j]
                    labels[i] = np.argmax(weighted_sum)
                else:
                    labels[i] = -1  # Abstain if no valid outputs
        
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        return labels
    
    def _compute_lf_weights(self) -> np.ndarray:
        """Compute weights for labeling functions based on their accuracy"""
        if not hasattr(self, 'lf_weights'):
            n_lfs = len(self.lfs)
            self.lf_weights = np.ones(n_lfs)
            
            # Use a small validation set to estimate LF accuracy
            val_size = min(1000, len(self.X) // 10)
            val_indices = np.random.choice(len(self.X), val_size, replace=False)
            
            lf_outputs = self.get_lf_outputs()
            for i in range(n_lfs):
                valid_mask = lf_outputs[val_indices, i] != -1
                if np.any(valid_mask):
                    accuracy = np.mean(
                        lf_outputs[val_indices, i][valid_mask] == self.y[val_indices][valid_mask]
                    )
                    self.lf_weights[i] = accuracy
                else:
                    self.lf_weights[i] = 0.0
            
            # Normalize weights
            self.lf_weights = self.lf_weights / np.sum(self.lf_weights)
        
        return self.lf_weights

    def get_label_matrix(self) -> np.ndarray:
        """Get the label matrix from all labeling functions"""
        if self.lf_outputs is None:
            self.get_labels()  # This will compute lf_outputs
        return self.lf_outputs
    
    def get_coverage(self) -> float:
        """Get the coverage of labeling functions"""
        if self.lf_outputs is None:
            self.get_labels()  # This will compute lf_outputs
        return np.mean(self.lf_outputs != -1)
    
    def get_accuracy(self) -> float:
        """Get the accuracy of labeling functions (requires ground truth)"""
        if self.y is None:
            raise ValueError("Ground truth labels required for accuracy calculation")
        
        if self.lf_outputs is None:
            self.get_labels()  # This will compute lf_outputs
        
        # For each sample, get the majority vote
        preds = self.get_labels()
        
        # Calculate accuracy only on samples with predictions
        mask = preds != -1
        if np.any(mask):
            return np.mean(preds[mask] == self.y[mask])
        return 0.0

    def evaluate_lfs(self) -> dict:
        """Evaluate performance of labeling functions"""
        if self.y is None:
            raise ValueError("Ground truth labels required for evaluation")
        
        metrics = {}
        for i, (name, lf_info) in enumerate(self.lfs.items()):
            preds = self.lf_outputs[:, i]
            mask = preds != -1
            
            if np.any(mask):
                accuracy = np.mean(preds[mask] == self.y[mask])
                coverage = np.mean(mask)
                metrics[name] = {
                    'accuracy': accuracy,
                    'coverage': coverage
                }
        
        return metrics 