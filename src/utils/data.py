import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

def add_label_noise(dataset: Dataset, noise_ratio: float) -> Dataset:
    """
    Add random label noise to a dataset.
    
    Args:
        dataset: PyTorch dataset
        noise_ratio: Ratio of labels to corrupt (0-1)
        
    Returns:
        Dataset with noisy labels
    """
    class NoisyDataset(Dataset):
        def __init__(self, dataset, noise_ratio):
            self.dataset = dataset
            self.noise_ratio = noise_ratio
            self.num_classes = 10  # For MNIST
            
            # Generate noisy labels
            self.noisy_labels = []
            for i in range(len(dataset)):
                if np.random.random() < noise_ratio:
                    # Generate a random label different from the original
                    original_label = dataset[i][1]
                    possible_labels = list(range(self.num_classes))
                    possible_labels.remove(original_label)
                    noisy_label = np.random.choice(possible_labels)
                    self.noisy_labels.append(noisy_label)
                else:
                    self.noisy_labels.append(dataset[i][1])
            
            self.noisy_labels = torch.tensor(self.noisy_labels)
        
        def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
            data, _ = self.dataset[index]
            return data, self.noisy_labels[index]
        
        def __len__(self) -> int:
            return len(self.dataset)
    
    return NoisyDataset(dataset, noise_ratio)

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Perform mixup data augmentation.
    
    Args:
        x: Input data
        y: Labels
        alpha: Mixup parameter
        
    Returns:
        Mixed inputs, mixed labels, original labels, and mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam 