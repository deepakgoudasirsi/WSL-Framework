import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.data_programming.labeling_functions import DataProgrammingDataset
from src.models.noise_robust import RobustCNN, RobustResNet

@dataclass
class WSLStrategy:
    """Base class for weak supervision strategies"""
    name: str
    weight: float = 1.0
    is_active: bool = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from the strategy"""
        raise NotImplementedError("Subclasses must implement predict method")

class DataProgrammingStrategy(WSLStrategy):
    """Data programming strategy using labeling functions"""
    def __init__(
        self,
        dataset: DataProgrammingDataset,
        aggregation_method: str = 'weighted_vote',
        weight: float = 1.0
    ):
        super().__init__(name='data_programming', weight=weight)
        self.dataset = dataset
        self.aggregation_method = aggregation_method
        self.labels = None
        self.train_indices = None
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 128,
        epochs: int = 100,
        lr: float = 0.001
    ):
        """Train the data programming strategy"""
        # Store training indices for later use
        self.train_indices = np.arange(len(X))
        # Get labels for training data
        self.labels = self.dataset.get_labels(
            aggregation_method=self.aggregation_method
        )
    
    def get_labels(self) -> np.ndarray:
        """Get labels from data programming"""
        if self.labels is None:
            self.labels = self.dataset.get_labels(
                aggregation_method=self.aggregation_method
            )
        return self.labels
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions for new data"""
        if self.labels is None:
            raise ValueError("Strategy not trained. Please call train() first.")
        
        # For validation/test data, we need to map back to training indices
        if len(X) != len(self.dataset.X):
            # This is validation/test data
            # Use the noise-robust strategy's predictions for these samples
            return np.full(len(X), -1)  # Return unknown labels
        
        # For training data, return the stored labels
        return self.labels

class NoiseRobustStrategy(WSLStrategy):
    """Noise-robust learning strategy"""
    def __init__(
        self,
        model_type: str = 'robust_cnn',
        loss_type: str = 'gce',
        weight: float = 1.0
    ):
        super().__init__(name='noise_robust', weight=weight)
        self.model_type = model_type
        self.loss_type = loss_type
        self.model = None
    
    def create_model(self, num_classes: int) -> nn.Module:
        """Create noise-robust model"""
        if self.model_type == 'robust_cnn':
            self.model = RobustCNN(
                num_classes=num_classes,
                loss_type=self.loss_type
            )
        else:
            self.model = RobustResNet(
                num_classes=num_classes,
                model_type=self.model_type,
                loss_type=self.loss_type
            )
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from the noise-robust model"""
        if self.model is None:
            raise ValueError("Model not created. Please train the strategy first.")
        
        with torch.no_grad():
            outputs = self.model(torch.from_numpy(X).float())
            preds = outputs.argmax(dim=1).numpy()
        return preds

class AdaptiveLearning:
    """Adaptive learning mechanism for combining strategies"""
    def __init__(self, strategies: List[WSLStrategy], weights: Optional[np.ndarray] = None):
        self.strategies = strategies
        self.weights = weights if weights is not None else np.ones(len(strategies))
        self.weights = self.weights / np.sum(self.weights)  # Normalize weights
        self.performance_history: Dict[str, List[float]] = {
            strategy.name: [] for strategy in strategies
        }
    
    def get_combined_labels(self, X: np.ndarray) -> np.ndarray:
        """Get combined labels from all strategies"""
        # Get predictions from each strategy
        strategy_preds = []
        for strategy in self.strategies:
            if not strategy.is_active:
                continue
            
            if isinstance(strategy, DataProgrammingStrategy):
                preds = strategy.get_labels()
            elif isinstance(strategy, NoiseRobustStrategy):
                if strategy.model is None:
                    raise ValueError(f"Model not created for strategy {strategy.name}")
                with torch.no_grad():
                    outputs = strategy.model(torch.from_numpy(X).float())
                    preds = outputs.argmax(dim=1).numpy()
            else:
                raise ValueError(f"Unknown strategy type: {type(strategy)}")
            
            strategy_preds.append(preds)
        
        # Stack predictions
        stacked_preds = np.stack(strategy_preds)
        
        # Apply weights
        weighted_labels = stacked_preds * self.weights[:, np.newaxis]
        
        # For each sample, get the weighted average prediction
        combined_labels = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            # Get non-zero predictions for this sample
            sample_preds = weighted_labels[:, i]
            valid_preds = sample_preds[sample_preds != 0]
            
            if len(valid_preds) > 0:
                # Round to nearest integer for classification
                combined_labels[i] = np.round(np.mean(valid_preds))
            else:
                # If no valid predictions, use -1 as unknown
                combined_labels[i] = -1
        
        return combined_labels.astype(int)
    
    def update_weights(self, X: np.ndarray, y: np.ndarray):
        """Update strategy weights based on performance"""
        accuracies = []
        for strategy in self.strategies:
            if not strategy.is_active:
                continue
            
            preds = strategy.predict(X)
            mask = preds != -1  # Only consider valid predictions
            if np.any(mask):
                acc = np.mean(preds[mask] == y[mask])
                accuracies.append(acc)
            else:
                accuracies.append(0.0)
        
        # Update weights based on accuracies
        accuracies = np.array(accuracies)
        self.weights = accuracies / np.sum(accuracies)

class ModelSelector:
    """Model selection based on multiple criteria"""
    def __init__(
        self,
        criteria: List[str] = ['accuracy', 'f1', 'robustness']
    ):
        self.criteria = criteria
        self.weights = [1.0] * len(criteria)
    
    def set_criteria_weights(self, weights: List[float]):
        """Set weights for different criteria"""
        if len(weights) != len(self.criteria):
            raise ValueError("Number of weights must match number of criteria")
        self.weights = weights
    
    def evaluate_model(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        noise_level: float = 0.0
    ) -> Dict[str, float]:
        """Evaluate model on multiple criteria"""
        with torch.no_grad():
            outputs = model(torch.from_numpy(X).float())
            preds = outputs.argmax(dim=1).numpy()
        
        metrics = {}
        
        if 'accuracy' in self.criteria:
            metrics['accuracy'] = accuracy_score(y, preds)
        
        if 'f1' in self.criteria:
            metrics['f1'] = f1_score(y, preds, average='weighted')
        
        if 'robustness' in self.criteria:
            # Add noise to inputs
            noisy_X = X + np.random.normal(0, noise_level, X.shape)
            with torch.no_grad():
                noisy_outputs = model(torch.from_numpy(noisy_X).float())
                noisy_preds = noisy_outputs.argmax(dim=1).numpy()
            metrics['robustness'] = accuracy_score(preds, noisy_preds)
        
        return metrics
    
    def select_model(
        self,
        models: List[nn.Module],
        X: np.ndarray,
        y: np.ndarray,
        noise_level: float = 0.0
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """Select best model based on weighted criteria"""
        best_score = -float('inf')
        best_model = None
        best_metrics = None
        
        for model in models:
            metrics = self.evaluate_model(model, X, y, noise_level)
            score = sum(
                metrics[c] * w
                for c, w in zip(self.criteria, self.weights)
            )
            
            if score > best_score:
                best_score = score
                best_model = model
                best_metrics = metrics
        
        return best_model, best_metrics

class UnifiedFramework:
    """Unified framework for weak supervision"""
    def __init__(
        self,
        strategies: List[WSLStrategy],
        model_selector: Optional[ModelSelector] = None
    ):
        self.strategies = strategies
        self.model_selector = model_selector or ModelSelector()
        self.weights = None
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_X: Optional[np.ndarray] = None,
        val_y: Optional[np.ndarray] = None,
        batch_size: int = 128,
        epochs: int = 100,
        lr: float = 0.001
    ) -> Dict[str, float]:
        """Train the framework"""
        # Use provided validation data or split from training data
        if val_X is None or val_y is None:
            n_samples = len(X)
            indices = np.random.permutation(n_samples)
            val_size = int(n_samples * 0.1)  # 10% validation
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
        else:
            X_train, y_train = X, y
            X_val, y_val = val_X, val_y
        
        # Train each strategy
        performances = {}
        for strategy in self.strategies:
            print(f"\nTraining {strategy.name}...")
            
            if isinstance(strategy, DataProgrammingStrategy):
                # Train data programming strategy
                strategy.train(
                    X=X_train,
                    y=y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    lr=lr
                )
                # Get predictions for validation set
                preds = strategy.predict(X_val)
                # Only evaluate on samples where we have predictions
                mask = preds != -1
                if np.any(mask):
                    performances[strategy.name] = accuracy_score(y_val[mask], preds[mask])
                else:
                    performances[strategy.name] = 0.0
            
            elif isinstance(strategy, NoiseRobustStrategy):
                # Train noise-robust model
                model = strategy.create_model(num_classes=len(np.unique(y)))
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                
                for epoch in range(epochs):
                    # Training loop
                    model.train()
                    for i in range(0, len(X_train), batch_size):
                        batch_X = torch.from_numpy(X_train[i:i+batch_size]).float()
                        batch_y = torch.from_numpy(y_train[i:i+batch_size]).long()
                        
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = nn.CrossEntropyLoss()(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                
                # Evaluate on validation set
                model.eval()
                with torch.no_grad():
                    outputs = model(torch.from_numpy(X_val).float())
                    preds = outputs.argmax(dim=1).numpy()
                    performances[strategy.name] = accuracy_score(y_val, preds)
            
            print(f"{strategy.name} validation accuracy: {performances[strategy.name]:.4f}")
        
        # Compute strategy weights
        self.weights = self.compute_weights(performances)
        print("\nStrategy weights:", self.weights)
        
        return performances
    
    def compute_weights(self, performances: Dict[str, float]) -> np.ndarray:
        """Compute weights for each strategy based on performance"""
        weights = np.array([performances[s.name] for s in self.strategies])
        return weights / np.sum(weights)  # Normalize weights
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from the unified framework"""
        if self.weights is None:
            raise ValueError("Framework not trained. Please call train() first.")
        
        # Get predictions from each strategy
        strategy_preds = []
        for strategy in self.strategies:
            preds = strategy.predict(X)
            strategy_preds.append(preds)
        
        # Stack predictions
        stacked_preds = np.stack(strategy_preds)
        
        # Apply weights
        weighted_labels = stacked_preds * self.weights[:, np.newaxis]
        
        # For each sample, get the weighted average prediction
        combined_labels = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            # Get non-zero predictions for this sample
            sample_preds = weighted_labels[:, i]
            valid_preds = sample_preds[sample_preds != 0]
            
            if len(valid_preds) > 0:
                # Round to nearest integer for classification
                combined_labels[i] = np.round(np.mean(valid_preds))
            else:
                # If no valid predictions, use -1 as unknown
                combined_labels[i] = -1
        
        return combined_labels.astype(int)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noise_level: float = 0.0
    ) -> Dict[str, float]:
        """Evaluate the unified framework"""
        preds = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, preds),
            'precision': precision_score(y, preds, average='weighted'),
            'recall': recall_score(y, preds, average='weighted'),
            'f1': f1_score(y, preds, average='weighted')
        }
        
        if self.model_selector is not None:
            # Evaluate robustness
            noisy_X = X + np.random.normal(0, noise_level, X.shape)
            noisy_preds = self.predict(noisy_X)
            metrics['robustness'] = accuracy_score(preds, noisy_preds)
        
        return metrics 