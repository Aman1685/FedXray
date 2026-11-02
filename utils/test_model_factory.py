
import os
import sys
import torch

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_DIR)

from model.model_factory import get_model


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_names = [
        "custom_cnn",
        "resnet18",
        "mobilenetv2",
        "alexnet"
    ]

    for name in model_names:

        print(f"\nTesting {name}...")

        model = get_model(
            model_name=name,
            num_classes=14,
            pretrained=False
        ).to(device)

        model.eval()

        # Use batch size 1 to limit GPU memory usage
        dummy_input = torch.randn(
            1, 1, 224, 224
        ).to(device)

        with torch.no_grad():
            output = model(dummy_input)

        parameters = sum(
            p.numel() for p in model.parameters()
        )

        print("Output shape:", output.shape)
        print("Parameters:", f"{parameters:,}")

        assert output.shape == (1, 14)

        del model, dummy_input, output

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"{name} passed!")

    print("\nAll four models passed!")


if __name__ == "__main__":
    main()
