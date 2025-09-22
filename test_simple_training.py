#!/usr/bin/env python3
"""
Simple training test to verify the fix
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models.baseline import SimpleCNN

def test_simple_training():
    """Test simple training with CIFAR-10"""
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataset = datasets.CIFAR10('data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Model
    model = SimpleCNN(num_classes=10)
    device = torch.device('cpu')
    model.to(device)
    
    # Optimizer with higher learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    print("Starting simple training test...")
    
    for epoch in range(5):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += target.size(0)
        
        test_acc = 100. * test_correct / test_total
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

if __name__ == '__main__':
    test_simple_training() 