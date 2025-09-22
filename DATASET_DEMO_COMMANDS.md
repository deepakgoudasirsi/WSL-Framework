# Dataset Demo Commands - CIFAR-10 and MNIST

## 🎯 DATASET CHARACTERISTICS DEMONSTRATION

### **1. SHOW DATASET BASIC INFORMATION**

```bash
# Show CIFAR-10 dataset information
python -c "
import torch
import torchvision
import torchvision.transforms as transforms

# Load CIFAR-10
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

print('CIFAR-10 Dataset Information:')
print('=' * 50)
print(f'Training samples: {len(trainset):,}')
print(f'Test samples: {len(testset):,}')
print(f'Total samples: {len(trainset) + len(testset):,}')
print(f'Number of classes: {len(trainset.classes)}')
print(f'Classes: {trainset.classes}')
print(f'Image size: {trainset[0][0].shape}')
print(f'Color channels: {trainset[0][0].shape[0]}')
print('=' * 50)
"
```

```bash
# Show MNIST dataset information
python -c "
import torch
import torchvision
import torchvision.transforms as transforms

# Load MNIST
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print('MNIST Dataset Information:')
print('=' * 50)
print(f'Training samples: {len(trainset):,}')
print(f'Test samples: {len(testset):,}')
print(f'Total samples: {len(trainset) + len(testset):,}')
print(f'Number of classes: {len(trainset.classes)}')
print(f'Classes: {trainset.classes}')
print(f'Image size: {trainset[0][0].shape}')
print(f'Color channels: {trainset[0][0].shape[0]}')
print('=' * 50)
"
```

### **2. SHOW IMAGE SIZE AND FORMAT**

```bash
# Demonstrate CIFAR-10 image format
python -c "
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load CIFAR-10
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Get sample images
sample1, label1 = trainset[0]
sample2, label2 = trainset[1000]

print('CIFAR-10 Image Format:')
print('=' * 40)
print(f'Sample 1 shape: {sample1.shape}')
print(f'Sample 1 data type: {sample1.dtype}')
print(f'Sample 1 value range: [{sample1.min():.3f}, {sample1.max():.3f}]')
print(f'Sample 1 label: {trainset.classes[label1]}')
print(f'Sample 2 shape: {sample2.shape}')
print(f'Sample 2 label: {trainset.classes[label2]}')
print('=' * 40)

# Show image dimensions
print(f'CIFAR-10: {sample1.shape[2]}x{sample1.shape[1]}x{sample1.shape[0]} RGB images')
"
```

```bash
# Demonstrate MNIST image format
python -c "
import torch
import torchvision
import torchvision.transforms as transforms

# Load MNIST
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Get sample images
sample1, label1 = trainset[0]
sample2, label2 = trainset[1000]

print('MNIST Image Format:')
print('=' * 40)
print(f'Sample 1 shape: {sample1.shape}')
print(f'Sample 1 data type: {sample1.dtype}')
print(f'Sample 1 value range: [{sample1.min():.3f}, {sample1.max():.3f}]')
print(f'Sample 1 label: {trainset.classes[label1]}')
print(f'Sample 2 shape: {sample2.shape}')
print(f'Sample 2 label: {trainset.classes[label2]}')
print('=' * 40)

# Show image dimensions
print(f'MNIST: {sample1.shape[2]}x{sample1.shape[1]}x{sample1.shape[0]} grayscale images')
"
```

### **3. SHOW NORMALIZATION PROCESS**

```bash
# Demonstrate normalization to [0,1] range
python -c "
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Load CIFAR-10 without normalization
trainset_raw = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Load CIFAR-10 with normalization
trainset_norm = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

print('Normalization Process:')
print('=' * 50)
print('Before Normalization:')
sample_raw = trainset_raw[0][0]
print(f'Value range: [{sample_raw.min():.3f}, {sample_raw.max():.3f}]')
print(f'Mean: {sample_raw.mean():.3f}')
print(f'Std: {sample_raw.std():.3f}')

print('\\nAfter Normalization:')
sample_norm = trainset_norm[0][0]
print(f'Value range: [{sample_norm.min():.3f}, {sample_norm.max():.3f}]')
print(f'Mean: {sample_norm.mean():.3f}')
print(f'Std: {sample_norm.std():.3f}')
print('=' * 50)
"
```

### **4. SHOW DATA AUGMENTATION**

```bash
# Demonstrate data augmentation techniques
python -c "
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Define augmentation transforms
augmentation_transforms = transforms.Compose([
    transforms.RandomRotation(15),  # Random rotation ±15°
    transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip 50% probability
    transforms.RandomCrop(32, padding=4),  # Random crop with padding
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 with augmentation
trainset_aug = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=augmentation_transforms)

print('Data Augmentation Techniques:')
print('=' * 50)
print('Applied augmentations:')
print('- Random rotation: ±15 degrees')
print('- Random horizontal flip: 50% probability')
print('- Random crop: 32x32 with 4px padding')
print('- Color jitter: brightness, contrast, saturation, hue')
print('- Normalization: mean=0.5, std=0.5')
print('=' * 50)

# Show augmented samples
print('\\nAugmented sample statistics:')
sample_aug = trainset_aug[0][0]
print(f'Shape: {sample_aug.shape}')
print(f'Value range: [{sample_aug.min():.3f}, {sample_aug.max():.3f}]')
print(f'Mean: {sample_aug.mean():.3f}')
print(f'Std: {sample_aug.std():.3f}')
"
```

### **5. SHOW BATCH PROCESSING**

```bash
# Demonstrate batch processing with configurable batch size
python -c "
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create data loader with batch size 128
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

print('Batch Processing Information:')
print('=' * 50)
print(f'Batch size: 128')
print(f'Number of batches: {len(trainloader)}')
print(f'Total samples: {len(trainset)}')

# Get first batch
dataiter = iter(trainloader)
images, labels = next(dataiter)

print(f'\\nFirst batch shape: {images.shape}')
print(f'Batch tensor shape: {images.shape}')
print(f'Labels shape: {labels.shape}')
print(f'Memory usage per batch: {images.element_size() * images.nelement() / 1024 / 1024:.2f} MB')
print('=' * 50)
"
```

### **6. SHOW DATASET COMPARISON**

```bash
# Compare CIFAR-10 and MNIST characteristics
python -c "
import torch
import torchvision
import torchvision.transforms as transforms

# Load both datasets
cifar_transform = transforms.Compose([transforms.ToTensor()])
mnist_transform = transforms.Compose([transforms.ToTensor()])

cifar_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)

print('Dataset Comparison: CIFAR-10 vs MNIST')
print('=' * 60)
print(f'{"Characteristic":<20} {"CIFAR-10":<15} {"MNIST":<15}')
print('-' * 60)
print(f'{"Image Size":<20} {"32x32x3":<15} {"28x28x1":<15}')
print(f'{"Color Channels":<20} {"3 (RGB)":<15} {"1 (Grayscale)":<15}')
print(f'{"Training Samples":<20} {len(cifar_trainset):<15} {len(mnist_trainset):<15}')
print(f'{"Classes":<20} {len(cifar_trainset.classes):<15} {len(mnist_trainset.classes):<15}')
print(f'{"Complexity":<20} {"High":<15} {"Low":<15}')
print('=' * 60)
"
```

### **7. SHOW MEMORY USAGE AND STORAGE**

```bash
# Show memory and storage requirements
python -c "
import os
import torch
import torchvision

# Calculate storage requirements
cifar_size = 32 * 32 * 3 * 4  # 32x32x3 RGB, 4 bytes per float32
mnist_size = 28 * 28 * 1 * 4  # 28x28x1 grayscale, 4 bytes per float32

print('Memory and Storage Requirements:')
print('=' * 50)
print(f'CIFAR-10 single image: {cifar_size} bytes ({cifar_size/1024:.2f} KB)')
print(f'MNIST single image: {mnist_size} bytes ({mnist_size/1024:.2f} KB)')
print(f'CIFAR-10 batch (128): {cifar_size * 128 / 1024 / 1024:.2f} MB')
print(f'MNIST batch (128): {mnist_size * 128 / 1024 / 1024:.2f} MB')
print('=' * 50)
"
```

### **8. SHOW DATA LOADING PIPELINE**

```bash
# Demonstrate the complete data loading pipeline
python -c "
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

print('Complete Data Loading Pipeline:')
print('=' * 50)

# 1. Define transforms
print('1. Define transforms:')
transforms_list = [
    'transforms.ToTensor()',
    'transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))',
    'transforms.RandomRotation(15)',
    'transforms.RandomHorizontalFlip(p=0.5)',
    'transforms.RandomCrop(32, padding=4)',
    'transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)'
]
for i, transform in enumerate(transforms_list, 1):
    print(f'   {i}. {transform}')

# 2. Load dataset
print('\\n2. Load dataset:')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
print(f'   Dataset loaded: {len(trainset)} samples')

# 3. Create data loader
print('\\n3. Create data loader:')
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
print(f'   DataLoader created: {len(trainloader)} batches')

# 4. Get sample batch
print('\\n4. Sample batch:')
dataiter = iter(trainloader)
images, labels = next(dataiter)
print(f'   Batch shape: {images.shape}')
print(f'   Labels shape: {labels.shape}')
print('=' * 50)
"
```

### **9. SHOW REAL-TIME DATA PROCESSING**

```bash
# Show real-time data processing with progress
python -c "
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

print('Real-time Data Processing Demo:')
print('=' * 50)

# Load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# Process batches with timing
start_time = time.time()
batch_count = 0
total_samples = 0

for batch_idx, (images, labels) in enumerate(trainloader):
    batch_count += 1
    total_samples += images.size(0)
    
    if batch_idx < 5:  # Show first 5 batches
        print(f'Batch {batch_idx + 1}: {images.shape}, Labels: {labels.shape}')
    
    if batch_idx >= 10:  # Stop after 10 batches for demo
        break

end_time = time.time()
processing_time = end_time - start_time

print(f'\\nProcessing Statistics:')
print(f'Batches processed: {batch_count}')
print(f'Total samples: {total_samples}')
print(f'Processing time: {processing_time:.2f} seconds')
print(f'Throughput: {total_samples/processing_time:.0f} samples/second')
print('=' * 50)
"
```

### **10. COMPREHENSIVE DATASET DEMO SCRIPT**

```bash
#!/bin/bash
echo "=== Dataset Demo: CIFAR-10 and MNIST ==="
echo ""

# 1. Show basic information
echo "1. Dataset Basic Information:"
python -c "
import torchvision
import torchvision.transforms as transforms

# CIFAR-10
cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
print(f'CIFAR-10: {len(cifar):,} samples, {cifar[0][0].shape[2]}x{cifar[0][0].shape[1]}x{cifar[0][0].shape[0]} RGB')

# MNIST
mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
print(f'MNIST: {len(mnist):,} samples, {mnist[0][0].shape[2]}x{mnist[0][0].shape[1]}x{mnist[0][0].shape[0]} grayscale')
"

# 2. Show normalization
echo ""
echo "2. Normalization Process:"
python -c "
import torchvision
import torchvision.transforms as transforms

cifar_raw = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
cifar_norm = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

raw_sample = cifar_raw[0][0]
norm_sample = cifar_norm[0][0]

print(f'Before normalization: [{raw_sample.min():.3f}, {raw_sample.max():.3f}]')
print(f'After normalization: [{norm_sample.min():.3f}, {norm_sample.max():.3f}]')
"

# 3. Show batch processing
echo ""
echo "3. Batch Processing:"
python -c "
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

dataiter = iter(trainloader)
images, labels = next(dataiter)
print(f'Batch shape: {images.shape}')
print(f'Labels shape: {labels.shape}')
print(f'Memory usage: {images.element_size() * images.nelement() / 1024 / 1024:.2f} MB')
"

echo ""
echo "=== Dataset Demo Complete ==="
```

## 🎯 **PRESENTATION TIPS**

### **What Each Command Demonstrates:**

1. **Basic Information**: Shows dataset sizes, classes, and image formats
2. **Image Format**: Demonstrates RGB vs grayscale, dimensions
3. **Normalization**: Shows before/after pixel value ranges
4. **Data Augmentation**: Lists all augmentation techniques applied
5. **Batch Processing**: Shows how data is batched for training
6. **Comparison**: Side-by-side comparison of CIFAR-10 vs MNIST
7. **Memory Usage**: Shows storage and memory requirements
8. **Pipeline**: Complete data loading process
9. **Real-time Processing**: Shows processing speed and throughput

### **Key Points to Emphasize:**

- **CIFAR-10**: 32x32x3 RGB images, complex, 10 classes
- **MNIST**: 28x28x1 grayscale images, simple, 10 classes
- **Normalization**: Pixel values normalized to [0,1] or [-1,1] range
- **Augmentation**: Multiple techniques for robustness
- **Batch Processing**: Configurable batch size (typically 128)
- **Memory Efficiency**: Optimized data loading and processing

### **Visual Demonstration:**

```bash
# Show sample images (if matplotlib is available)
python -c "
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

# Load and show sample images
cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

print('Sample images loaded successfully!')
print('CIFAR-10: Color images with complex patterns')
print('MNIST: Grayscale digit images')
"
```

These commands will help you demonstrate the dataset characteristics, normalization, and processing pipeline during your presentation! 🎯 