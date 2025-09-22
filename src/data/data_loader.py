import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Optional

class NoisyDataset(Dataset):
    """Dataset class for handling noisy labeled data"""
    
    def __init__(self, 
                 data: np.ndarray, 
                 labels: np.ndarray,
                 noise_type: str = 'symmetric',
                 noise_rate: float = 0.1):
        """
        Args:
            data: Input features
            labels: Target labels
            noise_type: Type of noise ('symmetric', 'asymmetric', 'instance')
            noise_rate: Probability of label noise
        """
        self.data = data
        self.clean_labels = labels
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.noisy_labels = self._add_noise(labels)
        
    def _add_noise(self, labels: np.ndarray) -> np.ndarray:
        """Add noise to labels based on specified noise type"""
        noisy_labels = labels.copy()
        n_samples = len(labels)
        n_noisy = int(n_samples * self.noise_rate)
        
        if self.noise_type == 'symmetric':
            # Randomly flip labels
            noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
            noisy_labels[noisy_indices] = np.random.randint(0, np.max(labels) + 1, n_noisy)
        elif self.noise_type == 'asymmetric':
            # Flip labels to similar classes
            noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
            noisy_labels[noisy_indices] = (labels[noisy_indices] + 1) % (np.max(labels) + 1)
            
        return noisy_labels
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.FloatTensor(self.data[idx]),
            torch.LongTensor([self.noisy_labels[idx]]),
            torch.LongTensor([self.clean_labels[idx]])
        )

def get_dataloader(data: np.ndarray,
                  labels: np.ndarray,
                  batch_size: int = 32,
                  noise_type: str = 'symmetric',
                  noise_rate: float = 0.1,
                  shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader for noisy labeled data
    
    Args:
        data: Input features
        labels: Target labels
        batch_size: Batch size for DataLoader
        noise_type: Type of noise to add
        noise_rate: Probability of label noise
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader object
    """
    dataset = NoisyDataset(data, labels, noise_type, noise_rate)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 