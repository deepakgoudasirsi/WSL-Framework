import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer class for noise-robust models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        save_dir: Path,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        
        # Move model to device
        self.model = self.model.to(device)
        if hasattr(self.model, 'model2'):
            self.model.model2 = self.model.model2.to(device)
        
        # Initialize metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self, epochs: int, early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        """Train the model."""
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch()
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}"
            )
            
            # Step the scheduler if it exists
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self._save_checkpoint(epoch, val_acc)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Plot training curves
        self.plot_curves()
        
        # Evaluate on test set
        self.evaluate()
        
        return history
    
    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Set epoch for models that support hybrid training
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(epoch)
        
        total_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        for inputs, targets in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # If the model supports co-teaching, use it
            if hasattr(self.model, 'use_co_teaching') and self.model.use_co_teaching:
                loss1, loss2 = self.model.co_teaching_step(inputs, targets, epoch)
                loss = loss1 + loss2
            elif hasattr(self.model, 'compute_loss'):
                outputs = self.model(inputs)
                # Check if compute_loss accepts epoch parameter
                import inspect
                sig = inspect.signature(self.model.compute_loss)
                if 'epoch' in sig.parameters:
                    result = self.model.compute_loss(outputs, targets, epoch)
                else:
                    result = self.model.compute_loss(outputs, targets)
                if isinstance(result, tuple):
                    loss, _ = result
                else:
                    loss = result
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracy
            with torch.no_grad():
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(self.train_loader), correct / total
    
    def _validate_epoch(self) -> tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                if hasattr(self.model, 'compute_loss'):
                    result = self.model.compute_loss(outputs, targets)
                    if isinstance(result, tuple):
                        loss, _ = result
                    else:
                        loss = result
                else:
                    loss = criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(self.val_loader), correct / total
    
    def _save_checkpoint(self, epoch: int, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc
        }
        if hasattr(self.model, 'model2'):
            checkpoint['model2_state_dict'] = self.model.model2.state_dict()
        
        torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pt')
    
    def plot_curves(self):
        """Plot training curves"""
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png')
        plt.close()
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\nEvaluating on test set...")
        
        # Load best model
        best_path = self.save_dir / 'best_model.pt'
        if best_path.exists():
            checkpoint = torch.load(best_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        test_metrics = self._validate_epoch()
        print(f"Test Loss: {test_metrics[0]:.4f}")
        print(f"Test Accuracy: {test_metrics[1]:.4f}")
        
        # Save test results
        results = {
            'test_loss': test_metrics[0],
            'test_accuracy': test_metrics[1],
            'best_epoch': self.best_epoch,
            'best_val_acc': self.best_val_acc
        }
        
        with open(self.save_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=4) 