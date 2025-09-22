import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models.noise_robust_model import NoiseRobustModel
from training.train import Trainer
from utils.data import add_label_noise

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path('experiments/noise_robust')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('data', train=False, transform=transform)
    
    # Split training data into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Add label noise to training data
    noise_ratio = 0.2  # 20% noisy labels
    train_dataset = add_label_noise(train_dataset, noise_ratio)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=0)
    
    # Create model
    model = NoiseRobustModel(
        model_type='simple_cnn',
        num_classes=10,
        loss_type='cross_entropy',
        beta=0.95,
        use_co_teaching=True,
        use_mixup=True,
        use_label_smoothing=True
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_dir=save_dir
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(epochs=50, early_stopping_patience=10)
    
    # Save training history
    torch.save(history, save_dir / 'training_history.pt')
    logger.info("Training completed!")

if __name__ == '__main__':
    main() 