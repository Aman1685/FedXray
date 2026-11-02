
import os
import sys
import torch

# Add the project root to Python's module search path
PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_DIR)

from model.custom_cnn import CustomCNN


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CustomCNN(num_classes=14).to(device)

    dummy_images = torch.randn(
        4, 1, 224, 224
    ).to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(dummy_images)

    print("Project directory:", PROJECT_DIR)
    print("Device:", device)
    print("Input shape:", dummy_images.shape)
    print("Output shape:", outputs.shape)

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("Total parameters:", total_parameters)

    assert outputs.shape == (4, 14)

    print("\nCustom CNN test passed!")


if __name__ == "__main__":
    main()
