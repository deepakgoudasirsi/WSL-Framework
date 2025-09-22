#!/usr/bin/env python3
"""
Minimal test to verify training works
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models.baseline import SimpleCNN

def minimal_test():
    """Minimal test with CIFAR-10"""
    
    print("Starting minimal test...")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Model
    model = SimpleCNN(num_classes=10)
    device = torch.device('cpu')
    model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    print("Model created successfully")
    
    # Single training step
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 1:  # Only do first batch
            break
            
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")
        print(f"Output shape: {output.shape}")
        print(f"Target shape: {target.shape}")
        
        # Check accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        total = target.size(0)
        acc = 100. * correct / total
        
        print(f"Accuracy: {acc:.2f}%")
        print("Training step completed successfully!")

if __name__ == '__main__':
    minimal_test() 