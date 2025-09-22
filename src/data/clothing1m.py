import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import pandas as pd
from PIL import Image
import numpy as np
from typing import Tuple, Dict, Any, Optional

class Clothing1MDataset(Dataset):
    """Clothing1M Dataset loader"""
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        noise_rate: float = 0.0,
        num_samples: Optional[int] = None
    ):
        """
        Args:
            root_dir (str): Directory with all the images
            split (str): One of ['train', 'val', 'test']
            transform (callable, optional): Optional transform to be applied on a sample
            noise_rate (float): Rate of noise to inject (0.0 to 1.0)
            num_samples (int, optional): Number of samples to use (for quick testing)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or self._get_default_transform()
        self.noise_rate = noise_rate
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Filter by split
        self.annotations = self.annotations[self.annotations['split'] == split]
        
        # Limit number of samples if specified
        if num_samples is not None:
            self.annotations = self.annotations.sample(n=num_samples, random_state=42)
        
        # Inject noise if specified
        if noise_rate > 0 and split == 'train':
            self._inject_noise()
    
    def _load_annotations(self) -> pd.DataFrame:
        """Load dataset annotations"""
        # Load the main annotation file
        annotations_path = os.path.join(self.root_dir, 'annotations', 'noisy_label_kv.txt')
        clean_annotations_path = os.path.join(self.root_dir, 'annotations', 'clean_label_kv.txt')
        
        # Load noisy labels
        noisy_labels = {}
        with open(annotations_path, 'r') as f:
            for line in f:
                img_path, label = line.strip().split()
                noisy_labels[img_path] = int(label)
        
        # Load clean labels if available
        clean_labels = {}
        if os.path.exists(clean_annotations_path):
            with open(clean_annotations_path, 'r') as f:
                for line in f:
                    img_path, label = line.strip().split()
                    clean_labels[img_path] = int(label)
        
        # Create DataFrame
        data = []
        for img_path, noisy_label in noisy_labels.items():
            clean_label = clean_labels.get(img_path, noisy_label)
            split = self._get_split(img_path)
            data.append({
                'image_path': img_path,
                'noisy_label': noisy_label,
                'clean_label': clean_label,
                'split': split
            })
        
        return pd.DataFrame(data)
    
    def _get_split(self, img_path: str) -> str:
        """Determine split based on image path"""
        if 'train' in img_path:
            return 'train'
        elif 'val' in img_path:
            return 'val'
        elif 'test' in img_path:
            return 'test'
        else:
            return 'train'  # Default to train
    
    def _inject_noise(self):
        """Inject additional noise into training set"""
        if self.noise_rate <= 0:
            return
        
        # Get unique classes
        classes = self.annotations['noisy_label'].unique()
        n_classes = len(classes)
        
        # Randomly select samples to corrupt
        n_samples = int(len(self.annotations) * self.noise_rate)
        corrupt_indices = np.random.choice(
            len(self.annotations),
            size=n_samples,
            replace=False
        )
        
        # Corrupt labels
        for idx in corrupt_indices:
            current_label = self.annotations.iloc[idx]['noisy_label']
            # Choose a different random label
            new_label = np.random.choice(
                [c for c in classes if c != current_label]
            )
            self.annotations.iloc[idx, self.annotations.columns.get_loc('noisy_label')] = new_label
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default transform for the dataset"""
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """Get a sample from the dataset"""
        row = self.annotations.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.root_dir, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, row['noisy_label'], row['clean_label']

def get_clothing1m_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    noise_rate: float = 0.0,
    num_samples: Optional[int] = None
) -> Dict[str, DataLoader]:
    """
    Get dataloaders for Clothing1M dataset
    
    Args:
        root_dir (str): Directory containing the dataset
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders
        noise_rate (float): Rate of noise to inject
        num_samples (int, optional): Number of samples to use
    
    Returns:
        Dict[str, DataLoader]: Dictionary containing dataloaders for train, val, and test sets
    """
    # Create datasets
    train_dataset = Clothing1MDataset(
        root_dir=root_dir,
        split='train',
        noise_rate=noise_rate,
        num_samples=num_samples
    )
    
    val_dataset = Clothing1MDataset(
        root_dir=root_dir,
        split='val',
        num_samples=num_samples
    )
    
    test_dataset = Clothing1MDataset(
        root_dir=root_dir,
        split='test',
        num_samples=num_samples
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    } 