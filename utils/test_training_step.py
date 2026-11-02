
import os
import sys
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_DIR)

from dataset.nih_dataset import NIHChestXrayDataset
from model.model_factory import get_model


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Training device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load a small batch of real NIH images
    train_csv = os.path.join(
        PROJECT_DIR, "dataset", "splits", "train.csv"
    )

    dataset = NIHChestXrayDataset(train_csv)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    images, labels = next(iter(loader))

    images = images.to(device)
    labels = labels.to(device)

    # Initialize Custom CNN
    model = get_model(
        model_name="custom_cnn",
        num_classes=14
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    # Forward pass
    model.train()

    optimizer.zero_grad()

    outputs = model(images)

    # Calculate loss
    loss = criterion(outputs, labels)

    # Backpropagation
    loss.backward()

    # Update model weights
    optimizer.step()

    print("\nTraining step results")
    print("---------------------")
    print("Input shape:", images.shape)
    print("Output shape:", outputs.shape)
    print("Label shape:", labels.shape)
    print("Loss:", loss.item())

    print("GPU memory allocated:",
          round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
          "MB" if device.type == "cuda" else "")

    print("\nActual training step passed!")


if __name__ == "__main__":
    main()
