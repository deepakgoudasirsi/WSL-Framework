#!/usr/bin/env python3
"""
Simple test for noise robustness fix
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models.baseline import MLP
from src.models.noise_robust import GCE, SCE

def test_loss_functions():
    """Test the fixed loss functions"""
    
    # Simple data
    outputs = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    
    print("Testing loss functions...")
    
    # Test GCE
    try:
        gce_loss = GCE(q=0.7)
        loss = gce_loss(outputs, targets)
        print(f"GCE loss: {loss.item():.4f}")
    except Exception as e:
        print(f"GCE error: {e}")
    
    # Test SCE
    try:
        sce_loss = SCE(alpha=0.1, beta=0.1)
        loss = sce_loss(outputs, targets)
        print(f"SCE loss: {loss.item():.4f}")
    except Exception as e:
        print(f"SCE error: {e}")
    
    # Test standard CE
    ce_loss = nn.CrossEntropyLoss()
    loss = ce_loss(outputs, targets)
    print(f"CE loss: {loss.item():.4f}")

def test_training_step():
    """Test a single training step"""
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Model
    model = MLP(input_size=784, num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Test different loss functions
    loss_functions = [
        ('CE', nn.CrossEntropyLoss()),
        ('GCE', GCE(q=0.7)),
        ('SCE', SCE(alpha=0.1, beta=0.1))
    ]
    
    for name, loss_fn in loss_functions:
        print(f"\nTesting {name} loss...")
        
        # Reset model
        for param in model.parameters():
            param.data.fill_(0.1)
        
        # Single training step
        try:
            data, target = next(iter(train_loader))
            data = data.view(-1, 784)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            
            print(f"Loss: {loss.item():.4f}")
            
            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() > 0:
                loss.backward()
                optimizer.step()
                print(f"Training step completed successfully")
            else:
                print(f"Loss is invalid: {loss.item()}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    test_loss_functions()
    test_training_step() 