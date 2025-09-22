#!/usr/bin/env python3
"""
Test script to verify warning suppression is working
"""

# Import warning suppression utility first
from src.utils.warning_suppression import apply_warning_suppression
apply_warning_suppression()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def test_warning_suppression():
    """Test that warnings are suppressed during PyTorch operations"""
    print("Testing warning suppression...")
    
    # Create a simple model
    model = nn.Linear(784, 10)
    
    # Create a simple dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load a small subset of MNIST
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Run a few training steps
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    print("Running training steps...")
    for i, (data, target) in enumerate(dataloader):
        if i >= 3:  # Only run 3 batches
            break
            
        optimizer.zero_grad()
        output = model(data.view(data.size(0), -1))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        print(f"Batch {i+1}: Loss = {loss.item():.4f}")
    
    print("Warning suppression test completed successfully!")
    print("If you didn't see any MallocStackLogging warnings, the suppression is working.")

if __name__ == '__main__':
    test_warning_suppression() 