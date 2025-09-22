import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import certifi
import os
os.environ['SSL_CERT_FILE'] = certifi.where()

import torch
import torchvision
import numpy as np
from typing import Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.data import add_label_noise

class NoisyDataset(Dataset):
    """Base class for handling noisy datasets"""
    def __init__(
        self,
        dataset: Dataset,
        noise_type: str = 'random',
        noise_level: float = 0.2,
        seed: int = 42
    ):
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.seed = seed
        self.noisy_labels = self._generate_noisy_labels()
        
    def _generate_noisy_labels(self) -> torch.Tensor:
        """Generate noisy labels based on specified noise type and level"""
        np.random.seed(self.seed)
        original_labels = torch.tensor([label for _, label in self.dataset])
        num_samples = len(original_labels)
        num_classes = len(torch.unique(original_labels))
        
        if self.noise_type == 'random':
            # Generate random noise
            mask = np.random.random(num_samples) < self.noise_level
            noisy_labels = original_labels.clone()
            for idx in np.where(mask)[0]:
                # Randomly select a different label
                possible_labels = list(range(num_classes))
                possible_labels.remove(original_labels[idx].item())
                noisy_labels[idx] = np.random.choice(possible_labels)
                
        elif self.noise_type == 'instance_dependent':
            # TODO: Implement instance-dependent noise
            raise NotImplementedError("Instance-dependent noise not implemented yet")
            
        return noisy_labels
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample with both clean and noisy labels"""
        image, _ = self.dataset[index]
        return image, self.noisy_labels[index]
    
    def __len__(self) -> int:
        return len(self.dataset)

def get_preprocessing_transforms(
    dataset_name: str,
    is_train: bool = True
) -> transforms.Compose:
    """Get appropriate preprocessing transforms for each dataset"""
    if dataset_name == 'cifar10':
        if is_train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            ])
            
    elif dataset_name == 'mnist':
        if is_train:
            return transforms.Compose([
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def load_dataset(
    dataset_name: str,
    noise_type: str = 'random',
    noise_level: float = 0.2,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """Load and preprocess the specified dataset"""
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get transforms
    train_transform = get_preprocessing_transforms(dataset_name, is_train=True)
    test_transform = get_preprocessing_transforms(dataset_name, is_train=False)
    
    # Load datasets
    if dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=test_transform
        )
        
    elif dataset_name == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=train_transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=test_transform
        )
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create noisy training dataset
    noisy_train_dataset = NoisyDataset(
        train_dataset,
        noise_type=noise_type,
        noise_level=noise_level,
        seed=seed
    )
    
    # Create data loaders
    train_loader = DataLoader(
        noisy_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False  # Disable pin_memory for CPU training
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False  # Disable pin_memory for CPU training
    )
    
    return {
        'train': train_loader,
        'test': test_loader
    }

def estimate_noise_level(
    dataset: Dataset,
    model: Optional[torch.nn.Module] = None
) -> float:
    """Estimate the noise level in the dataset"""
    # TODO: Implement noise level estimation
    raise NotImplementedError("Noise level estimation not implemented yet")

def add_noise_to_labels(labels: np.ndarray, noise_type: str, noise_rate: float, num_classes: int = 10) -> np.ndarray:
    """Add noise to labels based on the specified noise type and rate."""
    n_samples = len(labels)
    n_noisy = int(n_samples * noise_rate)
    
    if noise_type == 'random':
        # Randomly flip labels
        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
        noisy_labels = labels.copy()
        noisy_labels[noisy_indices] = np.random.randint(0, num_classes, n_noisy)
        return noisy_labels
    elif noise_type == 'symmetric':
        # Symmetric noise: flip to any other class with equal probability
        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
        noisy_labels = labels.copy()
        for idx in noisy_indices:
            possible_labels = list(range(num_classes))
            possible_labels.remove(labels[idx])
            noisy_labels[idx] = np.random.choice(possible_labels)
        return noisy_labels
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

def get_cifar10_dataloaders(
    batch_size: int = 128,
    noise_type: str = 'random',
    noise_rate: float = 0.0,
    labeled_ratio: float = 1.0
) -> Dict[str, DataLoader]:
    """Get CIFAR-10 dataloaders with optional label noise and semi-supervised split."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )
    
    # Split into labeled and unlabeled sets if needed
    if labeled_ratio < 1.0:
        labeled_dataset, unlabeled_dataset = create_semi_supervised_dataset(
            train_dataset,
            labeled_ratio=labeled_ratio
        )
        
        # Add noise to labeled data if specified
        if noise_rate > 0:
            labeled_dataset = add_label_noise(labeled_dataset, noise_rate)
        
        # Create dataloaders
        train_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return {
            'train': train_loader,
            'unlabeled': unlabeled_loader,
            'test': test_loader
        }
    else:
        # Add noise to training labels if specified
        if noise_rate > 0:
            train_labels = np.array(train_dataset.targets)
            noisy_labels = add_noise_to_labels(train_labels, noise_type, noise_rate)
            train_dataset.targets = noisy_labels
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return {
            'train': train_loader,
            'test': test_loader
        }

def get_mnist_dataloaders(
    batch_size: int = 128,
    noise_type: str = 'random',
    noise_rate: float = 0.0,
    labeled_ratio: float = 1.0
) -> Dict[str, DataLoader]:
    """Get MNIST dataloaders with optional label noise and semi-supervised split."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Split into labeled and unlabeled sets if needed
    if labeled_ratio < 1.0:
        labeled_dataset, unlabeled_dataset = create_semi_supervised_dataset(
            train_dataset,
            labeled_ratio=labeled_ratio
        )
        
        # Add noise to labeled data if specified
        if noise_rate > 0:
            labeled_dataset = add_label_noise(labeled_dataset, noise_rate)
        
        # Create dataloaders
        train_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return {
            'train': train_loader,
            'unlabeled': unlabeled_loader,
            'test': test_loader
        }
    else:
        # Add noise to training labels if specified
        if noise_rate > 0:
            train_labels = np.array(train_dataset.targets)
            noisy_labels = add_noise_to_labels(train_labels, noise_type, noise_rate)
            train_dataset.targets = noisy_labels
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return {
            'train': train_loader,
            'test': test_loader
        }

def get_clothing1m_dataloaders(
    batch_size: int = 128,
    num_workers: int = 4,
    num_samples: Optional[int] = None
) -> Dict[str, DataLoader]:
    """Get Clothing1M dataloaders"""
    # Note: This is a placeholder. You'll need to implement the actual data loading
    # based on your Clothing1M dataset structure.
    raise NotImplementedError("Clothing1M data loading not implemented yet")

def load_data_for_unified_framework(
    dataset: str,
    batch_size: int = 128,
    num_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load data for unified framework training"""
    if dataset == 'cifar10':
        loaders = get_cifar10_dataloaders(batch_size=batch_size, noise_rate=0.0)
    elif dataset == 'mnist':
        loaders = get_mnist_dataloaders(batch_size=batch_size, noise_rate=0.0)
    elif dataset == 'clothing1m':
        loaders = get_clothing1m_dataloaders(batch_size=batch_size, num_samples=num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Convert DataLoader to numpy arrays
    X_train, y_train = [], []
    for data, target in loaders['train']:
        X_train.append(data.numpy())
        y_train.append(target.numpy())
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    # If num_samples is specified, use the same number for test set
    if num_samples is not None:
        test_loader = loaders['test']
        X_test, y_test = [], []
        for data, target in test_loader:
            X_test.append(data.numpy())
            y_test.append(target.numpy())
            if len(X_test) * batch_size >= num_samples:
                break
        X_test = np.concatenate(X_test)[:num_samples]
        y_test = np.concatenate(y_test)[:num_samples]
    else:
        X_test, y_test = [], []
        for data, target in loaders['test']:
            X_test.append(data.numpy())
            y_test.append(target.numpy())
        X_test = np.concatenate(X_test)
        y_test = np.concatenate(y_test)

    return X_train, y_train, X_test, y_test

def create_semi_supervised_dataset(
    dataset: torch.utils.data.Dataset,
    labeled_ratio: float = 0.1,
    seed: int = 42
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Split dataset into labeled and unlabeled sets."""
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    labeled_size = int(len(dataset) * labeled_ratio)
    
    labeled_indices = indices[:labeled_size]
    unlabeled_indices = indices[labeled_size:]
    
    labeled_dataset = Subset(dataset, labeled_indices)
    unlabeled_dataset = Subset(dataset, unlabeled_indices)
    
    return labeled_dataset, unlabeled_dataset

if __name__ == '__main__':
    # Example usage
    cifar10_loaders = get_cifar10_dataloaders(
        batch_size=128,
        noise_type='random',
        noise_rate=0.0
    )
    
    mnist_loaders = get_mnist_dataloaders(
        batch_size=128,
        noise_type='random',
        noise_rate=0.0
    ) 