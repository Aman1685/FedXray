
import os
import sys
import torch

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_DIR)

from model.resnet18 import ResNet18


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ResNet18(
        num_classes=14,
        pretrained=False
    ).to(device)

    model.eval()

    dummy_images = torch.randn(
        2, 1, 224, 224
    ).to(device)

    with torch.no_grad():
        outputs = model(dummy_images)

    print("Device:", device)
    print("Input shape:", dummy_images.shape)
    print("Output shape:", outputs.shape)

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("Total parameters:", parameters)

    assert outputs.shape == (2, 14)

    print("\nResNet18 test passed!")


if __name__ == "__main__":
    main()
