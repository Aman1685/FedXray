
import os
import sys
import csv
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_DIR)

from dataset.nih_dataset import NIHChestXrayDataset
from model.model_factory import get_model


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    max_batches=None
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(loader):

        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("No samples were processed.")

    return total_loss / total_samples


def main():

    parser = argparse.ArgumentParser(
        description="Centralized NIH Chest X-ray training"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="custom_cnn",
        choices=[
            "custom_cnn",
            "resnet18",
            "mobilenetv2",
            "alexnet"
        ]
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the best checkpoint"
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nCentralized Training")
    print("====================")
    print("Model:", args.model)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Batch size:", args.batch_size)
    print("Additional epochs:", args.epochs)
    print("Learning rate:", args.lr)

    train_csv = os.path.join(
        PROJECT_DIR, "dataset", "splits", "train.csv"
    )

    val_csv = os.path.join(
        PROJECT_DIR, "dataset", "splits", "val.csv"
    )

    train_dataset = NIHChestXrayDataset(train_csv)
    val_dataset = NIHChestXrayDataset(val_csv)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    model = get_model(
        model_name=args.model,
        num_classes=14,
        pretrained=False
    ).to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Parameters:", f"{total_parameters:,}")

    positive_counts = train_dataset.labels.sum(axis=0)
    negative_counts = len(train_dataset) - positive_counts

    pos_weights = negative_counts / positive_counts
    pos_weights = pos_weights.clip(1.0, 20.0)

    pos_weights = torch.tensor(
        pos_weights,
        dtype=torch.float32,
        device=device
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights
    )

    print("\nClass weights")
    print("-------------")

    for disease, weight in zip(
        train_dataset.data.columns[4:18],
        pos_weights.tolist()
    ):
        print(f"{disease}: {weight:.2f}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    checkpoint_dir = os.path.join(
        PROJECT_DIR, "results", "checkpoints"
    )

    history_dir = os.path.join(
        PROJECT_DIR, "results", "centralized"
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{args.model}_centralized_best.pth"
    )

    history_path = os.path.join(
        history_dir,
        f"{args.model}_history.csv"
    )

    start_epoch = 0
    best_val_loss = float("inf")
    history = []

    # Resume from existing checkpoint
    if args.resume:

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        if checkpoint["model_name"] != args.model:
            raise ValueError("Checkpoint model does not match.")

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        best_val_loss = checkpoint["best_val_loss"]
        start_epoch = checkpoint["epoch"]

        if os.path.exists(history_path):
            with open(history_path, "r", newline="") as file:
                history = list(csv.DictReader(file))

        print("\nCheckpoint resumed!")
        print("Previous epoch:", start_epoch)
        print("Best validation loss:", best_val_loss)

    # Training loop
    for epoch in range(start_epoch, start_epoch + args.epochs):

        print(f"\nEpoch {epoch + 1}")
        print("-------------------------")

        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            max_batches=args.max_train_batches
        )

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_val_batches
        )

        print(f"Training loss:   {train_loss:.6f}")
        print(f"Validation loss: {val_loss:.6f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save({
                "epoch": epoch + 1,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "num_classes": 14
            }, checkpoint_path)

            print("Best checkpoint saved!")

        with open(
            history_path,
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "val_loss"
                ]
            )

            writer.writeheader()
            writer.writerows(history)

        if device.type == "cuda":
            print(
                "GPU memory allocated:",
                round(
                    torch.cuda.memory_allocated() / (1024 ** 2),
                    2
                ),
                "MB"
            )

    print("\nTraining completed!")
    print("Best validation loss:", best_val_loss)
    print("Checkpoint:", checkpoint_path)
    print("History:", history_path)


if __name__ == "__main__":
    main()
