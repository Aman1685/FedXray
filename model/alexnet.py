
import torch
import torch.nn as nn
from torchvision import models


class AlexNet(nn.Module):

    def __init__(self, num_classes=14, pretrained=False):

        super().__init__()

        weights = (
            models.AlexNet_Weights.DEFAULT
            if pretrained else None
        )

        self.model = models.alexnet(
            weights=weights
        )

        # Replace RGB input with grayscale input
        old_conv = self.model.features[0]

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )

        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(
                    old_conv.weight.mean(
                        dim=1,
                        keepdim=True
                    )
                )

                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)

        self.model.features[0] = new_conv

        # Replace final classifier
        in_features = self.model.classifier[6].in_features

        self.model.classifier[6] = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):

        return self.model(x)
