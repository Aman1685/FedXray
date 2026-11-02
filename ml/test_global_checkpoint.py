
import torch

from model.custom_cnn import CustomCNN


def main():

    checkpoint_path = (
        "results/checkpoints/fedavg_iid_smoke.pth"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load the saved federated checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    # Recreate the same CNN architecture
    model = CustomCNN(num_classes=14).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # Test using a dummy grayscale X-ray tensor
    dummy_image = torch.randn(
        1, 1, 224, 224
    ).to(device)

    with torch.no_grad():

        logits = model(dummy_image)

        probabilities = torch.sigmoid(logits)

    # Verify output
    assert logits.shape == (1, 14)
    assert torch.isfinite(logits).all()
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)

    print("\n========== CHECKPOINT TEST ==========")
    print("Checkpoint loaded successfully")
    print("Model:", checkpoint["model_name"])
    print("Communication round:", checkpoint["round"])
    print("Partition:", checkpoint["partition"])
    print("Output shape:", tuple(logits.shape))
    print("Probability range:",
          round(probabilities.min().item(), 4),
          "to",
          round(probabilities.max().item(), 4))

    print("\nSUCCESS: Global federated model is ready for inference!")


if __name__ == "__main__":
    main()
