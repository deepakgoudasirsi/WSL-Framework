import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

class Trainer:
    """Class for training and evaluating models"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 optimizer: optim.Optimizer = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer for training
            device: Device to run training on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters())
        else:
            self.optimizer = optimizer
            
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc='Training'):
            # Get data and move to device
            inputs, noisy_labels, clean_labels = batch
            inputs = inputs.to(self.device)
            noisy_labels = noisy_labels.squeeze().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, noisy_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                inputs, noisy_labels, clean_labels = batch
                inputs = inputs.to(self.device)
                clean_labels = clean_labels.squeeze().to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, clean_labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += clean_labels.size(0)
                correct += (predicted == clean_labels).sum().item()
                
        return total_loss / len(self.val_loader), correct / total
    
    def train(self, 
              num_epochs: int,
              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the model for specified number of epochs
        
        Args:
            num_epochs: Number of epochs to train
            save_path: Path to save best model
            
        Returns:
            Dictionary containing training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            
            # Train
            train_loss = self.train_epoch()
            history['train_loss'].append(train_loss)
            
            # Validate
            if self.val_loader is not None:
                val_loss, val_acc = self.validate()
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f'Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, '
                      f'Val Acc: {val_acc:.4f}')
                
                # Save best model
                if val_acc > best_val_acc and save_path is not None:
                    best_val_acc = val_acc
                    self.model.save(save_path)
            else:
                print(f'Train Loss: {train_loss:.4f}')
                
        return history 