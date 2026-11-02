
import torch
import torch.nn as nn
from torchvision import models


class ResNet18(nn.Module):

    def __init__(self, num_classes=14, pretrained=False):

        super().__init__()

        # Load ResNet18
        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained else None
        )

        self.model = models.resnet18(weights=weights)

        # Change first convolution to accept grayscale images
        old_conv = self.model.conv1

        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        if pretrained:
            # Convert pretrained RGB filters into grayscale filters
            with torch.no_grad():
                self.model.conv1.weight.copy_(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )

        # Change final layer to 14 disease outputs
        in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):

        return self.model(x)
