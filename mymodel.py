import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCIFAR10Net(nn.Module):
    def __init__(self, num_classes=10, use_batchnorm=True, use_dropout=False, activation='relu'):
        super(MyCIFAR10Net, self).__init__()
        # Example: 2 conv layers, pooling, batchnorm, dropout, fully connected
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25) if use_dropout else nn.Identity()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self._activate(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self._activate(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self._activate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def _activate(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'leakyrelu':
            return F.leaky_relu(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

# You can add more model variants or residual blocks here as needed.
