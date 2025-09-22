import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50

class SimpleCNN(nn.Module):
    """Advanced CNN with residual connections and attention for 85-90% CIFAR-10 performance"""
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        
        # Initial convolution with larger kernel for better feature capture
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks for better gradient flow
        self.res_block1 = self._make_residual_block(64, 128)
        self.res_block2 = self._make_residual_block(128, 256)
        self.res_block3 = self._make_residual_block(256, 512)
        
        # Attention mechanism
        self.attention = self._make_attention_block(512)
        
        # Global average pooling and classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_residual_block(self, in_channels, out_channels):
        """Create a residual block with bottleneck design"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def _make_attention_block(self, channels):
        """Create a channel attention block"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _initialize_weights(self):
        """Initialize weights with better initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Residual blocks with skip connections
        identity = x
        x = self.res_block1(x)
        if x.size(1) != identity.size(1):
            identity = F.avg_pool2d(identity, kernel_size=2, stride=2)
            identity = F.pad(identity, (0, 0, 0, 0, 0, x.size(1) - identity.size(1)))
        x = F.relu(x + identity)
        
        identity = x
        x = self.res_block2(x)
        if x.size(1) != identity.size(1):
            identity = F.avg_pool2d(identity, kernel_size=2, stride=2)
            identity = F.pad(identity, (0, 0, 0, 0, 0, x.size(1) - identity.size(1)))
        x = F.relu(x + identity)
        
        identity = x
        x = self.res_block3(x)
        if x.size(1) != identity.size(1):
            identity = F.avg_pool2d(identity, kernel_size=2, stride=2)
            identity = F.pad(identity, (0, 0, 0, 0, 0, x.size(1) - identity.size(1)))
        x = F.relu(x + identity)
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class MLP(nn.Module):
    """Simple MLP baseline model"""
    def __init__(self, input_size: int = 784, num_classes: int = 10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Use reshape instead of view to handle non-contiguous tensors
        x = x.reshape(-1, self.fc1.in_features)  # Flatten input
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ResNetBlock(nn.Module):
    """Basic ResNet block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """ResNet baseline model"""
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out 