#!/usr/bin/env python3
"""
Test GCE loss function
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models.baseline import SimpleCNN
from src.models.noise_robust import GCE

def test_gce_loss():
    """Test GCE loss function"""
    
    print("Testing GCE loss function...")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    
    # Model
    model = SimpleCNN(num_classes=10)
    device = torch.device('cpu')
    model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Test different loss functions
    loss_functions = [
        ('CE', nn.CrossEntropyLoss()),
        ('GCE_q=1.0', GCE(q=1.0)),
        ('GCE_q=0.7', GCE(q=0.7))
    ]
    
    for name, loss_fn in loss_functions:
        print(f"\nTesting {name}...")
        
        # Reset model
        for param in model.parameters():
            param.data.fill_(0.1)
        
        # Single training step
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 1:  # Only do first batch
                break
                
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            try:
                loss = loss_fn(output, target)
                print(f"Loss: {loss.item():.4f}")
                
                if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() > 0:
                    loss.backward()
                    optimizer.step()
                    
                    # Check accuracy
                    pred = output.argmax(dim=1, keepdim=True)
                    correct = pred.eq(target.view_as(pred)).sum().item()
                    total = target.size(0)
                    acc = 100. * correct / total
                    
                    print(f"Accuracy: {acc:.2f}%")
                    print(f"{name} training step completed successfully")
                else:
                    print(f"{name} loss is invalid: {loss.item()}")
                    
            except Exception as e:
                print(f"{name} error: {e}")

if __name__ == '__main__':
    test_gce_loss() 