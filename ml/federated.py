
import argparse
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

from dataset.nih_dataset import (
    NIHChestXrayDataset,
    DISEASE_LABELS
)
from model.custom_cnn import CustomCNN
from ml.clients.federated_client import train_client
from ml.aggregation.fedavg import fedavg_aggregate


def evaluate_global_model(
    model,
    loader,
    criterion,
    device,
    max_batches=None
):
    """Evaluate global model using the validation dataset."""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_probabilities = []
    all_labels = []

    with torch.no_grad():

        for batch_index, (images, labels) in enumerate(loader):

            if max_batches is not None and batch_index >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            probabilities = torch.sigmoid(logits.float())

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(
                labels.cpu().numpy()
            )

    if total_samples == 0:
        raise RuntimeError("No validation samples processed.")

    y_prob = np.concatenate(all_probabilities, axis=0)
    y_true = np.concatenate(all_labels, axis=0).astype(int)

    y_pred = (y_prob >= 0.5).astype(int)

    auc_values = []
    ap_values = []

    for index, disease in enumerate(DISEASE_LABELS):

        actual = y_true[:, index]
        probability = y_prob[:, index]

        if len(np.unique(actual)) == 2:
            auc_values.append(
                roc_auc_score(actual, probability)
            )

        if actual.sum() > 0:
            ap_values.append(
                average_precision_score(actual, probability)
            )

    metrics = {
        "macro_auc": float(np.mean(auc_values)),
        "micro_auc": float(
            roc_auc_score(y_true.ravel(), y_prob.ravel())
        ),
        "macro_ap": float(np.mean(ap_values)),
        "micro_ap": float(
            average_precision_score(
                y_true.ravel(),
                y_prob.ravel()
            )
        ),
        "micro_precision": float(
            precision_score(
                y_true.ravel(),
                y_pred.ravel(),
                zero_division=0
            )
        ),
        "micro_recall": float(
            recall_score(
                y_true.ravel(),
                y_pred.ravel(),
                zero_division=0
            )
        ),
        "micro_f1": float(
            f1_score(
                y_true.ravel(),
                y_pred.ravel(),
                zero_division=0
            )
        )
    }

    validation_loss = total_loss / total_samples

    return validation_loss, metrics, total_samples


def run_federated_training(args):

    # Reproducibility: use the same seed for every experiment
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Random seed:", args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n========== FEDERATED TRAINING ==========")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Dataset paths
    train_csv = os.path.join(
        "dataset", "splits", "train.csv"
    )

    val_csv = os.path.join(
        "dataset", "splits", "val.csv"
    )

    partition_dir = os.path.join(
        "dataset", "splits", args.partition
    )

    # Load datasets
    train_dataset = NIHChestXrayDataset(train_csv)
    val_dataset = NIHChestXrayDataset(val_csv)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Calculate global class weights using training data only
    positive_counts = train_dataset.labels.sum(axis=0)
    negative_counts = len(train_dataset) - positive_counts

    pos_weights = negative_counts / np.maximum(
        positive_counts, 1
    )

    pos_weights = np.clip(pos_weights, 1.0, 20.0)

    pos_weights = torch.tensor(
        pos_weights,
        dtype=torch.float32,
        device=device
    )

    print("\nGlobal class weights calculated.")

    # Same weighted BCE objective for all clients and validation
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights
    )

    # Initialize a fresh global model
    global_model = CustomCNN(
        num_classes=14
    ).to(device)

    checkpoint_dir = "results/checkpoints"
    history_dir = "results/federated"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        args.checkpoint_name
    )

    history_path = os.path.join(
        history_dir,
        f"{args.partition}_fedavg_history.csv"
    )

    best_val_loss = float("inf")
    history = []

    for round_num in range(1, args.rounds + 1):

        print(
            f"\n========== ROUND {round_num}/{args.rounds} =========="
        )

        client_states = []
        client_sizes = []
        client_losses = []

        # Train clients sequentially
        for client_id in range(1, args.num_clients + 1):

            csv_file = os.path.join(
                partition_dir,
                f"client_{client_id}.csv"
            )

            if not os.path.isfile(csv_file):
                raise FileNotFoundError(
                    f"Client dataset not found: {csv_file}"
                )

            local_state, num_samples, loss = train_client(
                global_model=global_model,
                csv_file=csv_file,
                client_id=client_id,
                device=device,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                max_batches=args.max_batches,
                pos_weight=pos_weights
            )

            client_states.append(local_state)
            client_sizes.append(num_samples)
            client_losses.append(loss)

        # Aggregate client model weights
        aggregated_state = fedavg_aggregate(
            client_states,
            client_sizes
        )

        global_model.load_state_dict(aggregated_state)

        weighted_train_loss = sum(
            loss * size
            for loss, size in zip(client_losses, client_sizes)
        ) / sum(client_sizes)

        # Evaluate global model
        val_loss, metrics, val_samples = evaluate_global_model(
            model=global_model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_val_batches
        )

        print("\n========== ROUND RESULTS ==========")
        print("Round:", round_num)
        print("Training samples:", sum(client_sizes))
        print("Validation samples:", val_samples)
        print(f"Weighted training loss: {weighted_train_loss:.6f}")
        print(f"Validation loss: {val_loss:.6f}")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Save history
        row = {
            "round": round_num,
            "partition": args.partition,
            "train_samples": sum(client_sizes),
            "validation_samples": val_samples,
            "train_loss": weighted_train_loss,
            "val_loss": val_loss,
            **metrics
        }

        history.append(row)

        # Save the best global model based on validation loss
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_name": "custom_cnn",
                    "seed": args.seed,
                    "model_state_dict": global_model.state_dict(),
                    "round": round_num,
                    "num_clients": args.num_clients,
                    "partition": args.partition,
                    "algorithm": "FedAvg",
                    "local_epochs": args.local_epochs,
                    "batch_size": args.batch_size,
                    "client_samples": client_sizes,
                    "client_losses": client_losses,
                    "weighted_training_loss": weighted_train_loss,
                    "validation_loss": val_loss,
                    "validation_metrics": metrics,
                    "num_classes": 14,
                    "best_round": round_num
                },
                checkpoint_path
            )

            print("\nBest global checkpoint saved!")

        with open(
            history_path,
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(history[0].keys())
            )

            writer.writeheader()
            writer.writerows(history)

        print("History saved:", history_path)

        del client_states
        del aggregated_state

        if device.type == "cuda":
            print(
                "GPU memory allocated:",
                round(
                    torch.cuda.memory_allocated() / (1024 ** 2),
                    2
                ),
                "MB"
            )

    print("\n========== FEDERATED TRAINING COMPLETED ==========")
    print("Best validation loss:", best_val_loss)
    print("Global checkpoint:", checkpoint_path)
    print("Experiment history:", history_path)


def main():

    parser = argparse.ArgumentParser(
        description="FedXray FedAvg training"
    )

    parser.add_argument(
        "--partition",
        type=str,
        default="clients_iid",
        choices=[
            "clients_iid",
            "clients_noniid",
            "clients_noniid_balanced"
        ]
    )

    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None
    )

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="fedavg_custom_cnn_iid_seed42_best.pth"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    if args.num_clients < 1:
        parser.error("--num-clients must be positive")

    if args.rounds < 1 or args.local_epochs < 1:
        parser.error("--rounds and --local-epochs must be positive")

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")

    if args.max_val_batches is not None and args.max_val_batches < 1:
        parser.error("--max-val-batches must be positive")

    run_federated_training(args)


if __name__ == "__main__":
    main()
