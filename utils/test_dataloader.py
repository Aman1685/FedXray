
import os
import sys
import torch

from torch.utils.data import DataLoader

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_DIR)

from dataset.nih_dataset import NIHChestXrayDataset


def main():

    print("\nTesting NIH Chest X-ray DataLoader")

    train_csv = os.path.join(
        PROJECT_DIR,
        "dataset",
        "splits",
        "train.csv"
    )

    val_csv = os.path.join(
        PROJECT_DIR,
        "dataset",
        "splits",
        "val.csv"
    )

    test_csv = os.path.join(
        PROJECT_DIR,
        "dataset",
        "splits",
        "test.csv"
    )

    # Load datasets
    train_dataset = NIHChestXrayDataset(train_csv)
    val_dataset = NIHChestXrayDataset(val_csv)
    test_dataset = NIHChestXrayDataset(test_csv)

    # Small batch size for initial testing
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # Fetch one batch
    images, labels = next(iter(train_loader))

    print("\nBatch verification")
    print("------------------")

    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)

    print("Image data type:", images.dtype)
    print("Label data type:", labels.dtype)

    print("Minimum pixel value:", images.min().item())
    print("Maximum pixel value:", images.max().item())

    print("Sample labels:", labels[0].tolist())

    print("\nDataset sizes")
    print("-------------")
    print("Training:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Testing:", len(test_dataset))

    # Final checks
    assert images.shape == (2, 1, 224, 224)
    assert labels.shape == (2, 14)

    assert images.dtype == torch.float32
    assert labels.dtype == torch.float32

    assert torch.all((labels == 0) | (labels == 1))

    print("\nDataLoader test passed!")


if __name__ == "__main__":
    main()
