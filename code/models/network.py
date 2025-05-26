"""Neural network architecture for WiFi-based Material Identification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from typing import Optional, Dict, Any

from config import LABELS, DROPOUT


class MaterialIdentificationNetwork(nn.Module):
    """Neural network for material identification using WiFi CSI data."""
    
    def __init__(self, num_classes: int = len(LABELS), dropout: float = DROPOUT):
        """Initialize the network.
        
        Args:
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        self.num_classes = num_classes
        
        # 2D Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=1, stride=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=(1, 1))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=(1, 1))
        
        # 1D Convolutional layers
        self.c1d = nn.Conv1d(1, 64, kernel_size=3, stride=2)
        self.c2d = nn.Conv1d(64, 128, kernel_size=3, stride=2)
        self.c3d = nn.Conv1d(128, 256, kernel_size=3, stride=2)
        
        # ResNet18 backbone (modified for our use case)
        self.resnet = resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        
        # Fully connected layers
        self.fc1 = nn.Linear(50944, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.fc4 = nn.Linear(num_classes * 2, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        self.leaky_relu = nn.LeakyReLU()
        
        # Freeze ResNet parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits
        """
        # Process through 2D conv layers
        x_2d = self.leaky_relu(self.conv1(x))
        x_2d = self.leaky_relu(self.conv2(x_2d))
        x_2d = self.leaky_relu(self.conv3(x_2d))
        x_2d = x_2d.view(x_2d.size(0), -1)  # Flatten
        
        # Process through 1D conv layers
        x_1d = x.squeeze(1)  # Remove channel dim for 1D conv
        x_1d = self.leaky_relu(self.c1d(x_1d))
        x_1d = self.leaky_relu(self.c2d(x_1d))
        x_1d = self.leaky_relu(self.c3d(x_1d))
        x_1d = x_1d.view(x_1d.size(0), -1)  # Flatten
        
        # Process through ResNet
        x_resnet = self.resnet(x)
        
        # Concatenate features
        x_combined = torch.cat([x_2d, x_1d], dim=1)
        x_combined = self.leaky_relu(self.fc1(x_combined))
        x_combined = self.dropout(x_combined)
        x_combined = self.leaky_relu(self.fc2(x_combined))
        x_combined = self.dropout(x_combined)
        x_combined = self.fc3(x_combined)
        
        # Final combination with ResNet features
        x_final = torch.cat([x_combined, x_resnet], dim=1)
        x_final = self.fc4(x_final)
        
        return x_final
    
    def compute_loss(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the cross-entropy loss.
        
        Args:
            preds: Model predictions (logits)
            labels: Ground truth labels
            
        Returns:
            Loss value
        """
        return F.cross_entropy(preds, labels)


def create_model(device: str = "cuda" if torch.cuda.is_available() else "cpu",
                num_classes: int = len(LABELS),
                dropout: float = DROPOUT) -> MaterialIdentificationNetwork:
    """Create and initialize the model.
    
    Args:
        device: Device to create the model on
        num_classes: Number of output classes
        dropout: Dropout probability
        
    Returns:
        Initialized model
    """
    model = MaterialIdentificationNetwork(
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    return model
