
import os
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from dataset.nih_dataset import (
    NIHChestXrayDataset,
    DISEASE_LABELS
)
from model.model_factory import get_model


CHECKPOINT_FILES = [
    "custom_cnn_centralized_best.pth",

    "fedavg_custom_cnn_iid_seed42_best.pth",
    "fedavg_custom_cnn_noniid_seed42_best.pth",
    "fedavg_custom_cnn_noniid_balanced_seed42_best.pth",

    "fedprox_custom_cnn_clients_iid_seed42_best.pth",
    "fedprox_custom_cnn_clients_noniid_seed42_best.pth",
    "fedprox_custom_cnn_clients_noniid_balanced_seed42_best.pth",
]


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def safe_ap(y_true, y_prob):
    if np.sum(y_true) == 0:
        return np.nan
    return average_precision_score(y_true, y_prob)


def evaluate_model(
    checkpoint_path,
    test_loader,
    device,
    threshold
):
    filename = checkpoint_path.name

    print("\n" + "=" * 70)
    print("Evaluating:", filename)
    print("=" * 70)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    if checkpoint.get("model_name") != "custom_cnn":
        raise ValueError(
            f"Unexpected model name in {filename}: "
            f"{checkpoint.get('model_name')}"
        )

    if checkpoint.get("num_classes") != len(DISEASE_LABELS):
        raise ValueError(
            f"Unexpected number of classes in {filename}"
        )

    model = get_model(
        model_name="custom_cnn",
        num_classes=len(DISEASE_LABELS),
        pretrained=False
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_probabilities = []
    all_labels = []

    with torch.inference_mode():

        for batch_index, (images, labels) in enumerate(test_loader):

            images = images.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                logits = model(images)

            probabilities = torch.sigmoid(logits.float())

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

            if (batch_index + 1) % 200 == 0:
                print(
                    f"Processed batches: {batch_index + 1}"
                )

    y_prob = np.concatenate(
        all_probabilities,
        axis=0
    )

    y_true = np.concatenate(
        all_labels,
        axis=0
    ).astype(np.int32)

    y_pred = (
        y_prob >= threshold
    ).astype(np.int32)

    # Overall multi-label metrics
    exact_match = accuracy_score(y_true, y_pred)

    label_accuracy = accuracy_score(
        y_true.ravel(),
        y_pred.ravel()
    )

    hamming = hamming_loss(y_true, y_pred)

    macro_auc_values = [
        safe_auc(y_true[:, i], y_prob[:, i])
        for i in range(len(DISEASE_LABELS))
    ]

    macro_ap_values = [
        safe_ap(y_true[:, i], y_prob[:, i])
        for i in range(len(DISEASE_LABELS))
    ]

    macro_auc = np.nanmean(macro_auc_values)
    macro_ap = np.nanmean(macro_ap_values)

    micro_auc = roc_auc_score(
        y_true.ravel(),
        y_prob.ravel()
    )

    micro_ap = average_precision_score(
        y_true.ravel(),
        y_prob.ravel()
    )

    micro_precision = precision_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    micro_recall = recall_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # Per-disease metrics
    disease_rows = []

    for index, disease in enumerate(DISEASE_LABELS):

        actual = y_true[:, index]
        predicted = y_pred[:, index]
        probability = y_prob[:, index]

        tn, fp, fn, tp = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1]
        ).ravel()

        disease_rows.append({
            "Disease": disease,
            "Positive_Count": int(actual.sum()),
            "Prevalence": float(actual.mean()),
            "Accuracy": float(
                accuracy_score(actual, predicted)
            ),
            "ROC_AUC": safe_auc(actual, probability),
            "Average_Precision": safe_ap(
                actual,
                probability
            ),
            "Precision": precision_score(
                actual,
                predicted,
                zero_division=0
            ),
            "Recall": recall_score(
                actual,
                predicted,
                zero_division=0
            ),
            "F1_Score": f1_score(
                actual,
                predicted,
                zero_division=0
            ),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        })

    summary = {
        "Checkpoint": filename,
        "Algorithm": checkpoint.get("algorithm", "centralized"),
        "Partition": checkpoint.get("partition", "centralized"),
        "Seed": checkpoint.get("seed", ""),
        "Best_Round": checkpoint.get(
            "best_round",
            checkpoint.get("epoch", "")
        ),
        "Threshold": threshold,
        "Test_Samples": len(y_true),
        "Exact_Match_Accuracy": exact_match,
        "Per_Label_Accuracy": label_accuracy,
        "Hamming_Loss": hamming,
        "Macro_ROC_AUC": macro_auc,
        "Micro_ROC_AUC": micro_auc,
        "Macro_AP": macro_ap,
        "Micro_AP": micro_ap,
        "Macro_Precision": macro_precision,
        "Micro_Precision": micro_precision,
        "Macro_Recall": macro_recall,
        "Micro_Recall": micro_recall,
        "Macro_F1": macro_f1,
        "Micro_F1": micro_f1
    }

    print("\nOverall test results")
    print("--------------------")

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.6f}")
        else:
            print(f"{key:25s}: {value}")

    # Release model memory before loading the next checkpoint
    del model
    del checkpoint

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary, disease_rows


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate all FedXray checkpoints on the official test set"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )

    args = parser.parse_args()

    if not 0 < args.threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if args.batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nFedXray: Official Test Evaluation")
    print("=================================")
    print("Device:", device)
    print("Batch size:", args.batch_size)
    print("Threshold:", args.threshold)

    test_csv = (
        PROJECT_DIR
        / "dataset"
        / "splits"
        / "test.csv"
    )

    checkpoint_dir = (
        PROJECT_DIR
        / "results"
        / "checkpoints"
    )

    output_dir = (
        PROJECT_DIR
        / "results"
        / "evaluation"
        / "official_test"
    )

    os.makedirs(output_dir, exist_ok=True)

    if not test_csv.is_file():
        raise FileNotFoundError(test_csv)

    # Verify all checkpoints before starting evaluation
    for filename in CHECKPOINT_FILES:
        path = checkpoint_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint: {path}"
            )

    # Load official test dataset only once
    test_dataset = NIHChestXrayDataset(str(test_csv))

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    print("\nOfficial test images:", len(test_dataset))

    summary_rows = []

    for filename in CHECKPOINT_FILES:

        checkpoint_path = checkpoint_dir / filename

        summary, disease_rows = evaluate_model(
            checkpoint_path,
            test_loader,
            device,
            args.threshold
        )

        summary_rows.append(summary)

        disease_path = (
            output_dir
            / f"{checkpoint_path.stem}_disease_metrics.csv"
        )

        with open(
            disease_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=disease_rows[0].keys()
            )

            writer.writeheader()
            writer.writerows(disease_rows)

        print("\nSaved per-disease metrics:", disease_path)

    # Save combined comparison
    summary_path = output_dir / "all_models_summary.csv"

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 70)
    print("ALL CHECKPOINTS EVALUATED")
    print("=" * 70)
    print("Summary:", summary_path)
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
