#!/usr/bin/env python3
"""
Debug script for noise robustness testing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models.baseline import MLP
from src.models.noise_robust import SCE, GCE

def test_mnist_training():
    """Test MNIST training with different loss functions"""
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Test different configurations
    configs = [
        {'loss_type': 'ce', 'lr': 0.001, 'name': 'CrossEntropy'},
        {'loss_type': 'sce', 'lr': 0.001, 'name': 'SCE_0.001'},
        {'loss_type': 'sce', 'lr': 0.01, 'name': 'SCE_0.01'},
        {'loss_type': 'gce', 'lr': 0.001, 'name': 'GCE_0.001'},
        {'loss_type': 'gce', 'lr': 0.01, 'name': 'GCE_0.01'},
    ]
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Testing: {config['name']}")
        print(f"{'='*50}")
        
        # Create model
        model = MLP(input_size=784, num_classes=10)
        device = torch.device('cpu')
        model.to(device)
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        
        # Create loss function
        if config['loss_type'] == 'ce':
            criterion = nn.CrossEntropyLoss()
        elif config['loss_type'] == 'sce':
            criterion = SCE(alpha=0.1, beta=1.0)
        elif config['loss_type'] == 'gce':
            criterion = GCE(q=0.7)
        
        # Training loop
        for epoch in range(5):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                data = data.view(-1, 784)  # Flatten for MLP
                
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
                    data = data.view(-1, 784)  # Flatten for MLP
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    test_correct += pred.eq(target.view_as(pred)).sum().item()
                    test_total += target.size(0)
            
            test_acc = 100. * test_correct / test_total
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

if __name__ == '__main__':
    test_mnist_training() 