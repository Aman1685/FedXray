
import os
import sys
import csv
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_DIR)

from dataset.nih_dataset import (
    NIHChestXrayDataset,
    DISEASE_LABELS
)

from model.model_factory import get_model


def evaluate():

    parser = argparse.ArgumentParser(
        description="Evaluate centralized NIH chest X-ray model"
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

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nNIH Chest X-ray Evaluation")
    print("==========================")
    print("Model:", args.model)
    print("Device:", device)
    print("Threshold:", args.threshold)

    # Validation dataset
    val_csv = os.path.join(
        PROJECT_DIR,
        "dataset",
        "splits",
        "val.csv"
    )

    val_dataset = NIHChestXrayDataset(val_csv)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Load model
    model = get_model(
        model_name=args.model,
        num_classes=14,
        pretrained=False
    ).to(device)

    checkpoint_path = os.path.join(
        PROJECT_DIR,
        "results",
        "checkpoints",
        f"{args.model}_centralized_best.pth"
    )

    if not os.path.isfile(checkpoint_path):
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

    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Checkpoint validation loss:",
          checkpoint["best_val_loss"])

    # Collect predictions and actual labels
    all_probabilities = []
    all_labels = []

    with torch.no_grad():

        for batch_index, (images, labels) in enumerate(val_loader):

            images = images.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                logits = model(images)

            probabilities = torch.sigmoid(
                logits.float()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

            if (batch_index + 1) % 100 == 0:
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
    ).astype(int)

    y_pred = (
        y_prob >= args.threshold
    ).astype(int)

    print("\nEvaluation samples:", len(y_true))
    print("Prediction shape:", y_prob.shape)
    print("Label shape:", y_true.shape)

    # Output directories
    output_dir = os.path.join(
        PROJECT_DIR,
        "results",
        "evaluation",
        args.model
    )

    confusion_dir = os.path.join(
        output_dir,
        "confusion_matrices"
    )

    os.makedirs(
        confusion_dir,
        exist_ok=True
    )

    # Per-disease metrics
    rows = []

    for index, disease in enumerate(DISEASE_LABELS):

        actual = y_true[:, index]
        predicted = y_pred[:, index]
        probability = y_prob[:, index]

        tn, fp, fn, tp = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1]
        ).ravel()

        # ROC-AUC is undefined if only one class exists
        if len(np.unique(actual)) == 2:
            auc = roc_auc_score(
                actual,
                probability
            )
        else:
            auc = np.nan

        # Average Precision
        if actual.sum() > 0:
            ap = average_precision_score(
                actual,
                probability
            )
        else:
            ap = np.nan

        precision = precision_score(
            actual,
            predicted,
            zero_division=0
        )

        recall = recall_score(
            actual,
            predicted,
            zero_division=0
        )

        f1 = f1_score(
            actual,
            predicted,
            zero_division=0
        )

        rows.append({
            "Disease": disease,
            "Positive_Count": int(actual.sum()),
            "ROC_AUC": auc,
            "Average_Precision": ap,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        })

        # Save confusion matrix
        cm = np.array([
            [tn, fp],
            [fn, tp]
        ])

        plt.figure(figsize=(5, 4))

        plt.imshow(cm, interpolation="nearest")

        plt.title(f"Confusion Matrix - {disease}")
        plt.colorbar()

        plt.xticks([0, 1], ["Negative", "Positive"])
        plt.yticks([0, 1], ["Negative", "Positive"])

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        for row_index in range(2):
            for col_index in range(2):
                plt.text(
                    col_index,
                    row_index,
                    str(cm[row_index, col_index]),
                    ha="center",
                    va="center"
                )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                confusion_dir,
                f"{disease}.png"
            ),
            dpi=150
        )

        plt.close()

    # Save disease metrics
    metrics_path = os.path.join(
        output_dir,
        "disease_metrics.csv"
    )

    with open(
        metrics_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    # Macro metrics: average across valid diseases
    auc_values = [
        row["ROC_AUC"]
        for row in rows
        if not np.isnan(row["ROC_AUC"])
    ]

    ap_values = [
        row["Average_Precision"]
        for row in rows
        if not np.isnan(row["Average_Precision"])
    ]

    macro_auc = np.mean(auc_values)
    macro_ap = np.mean(ap_values)

    # Micro metrics: combine all disease decisions
    micro_auc = roc_auc_score(
        y_true.ravel(),
        y_prob.ravel()
    )

    micro_ap = average_precision_score(
        y_true.ravel(),
        y_prob.ravel()
    )

    micro_precision = precision_score(
        y_true.ravel(),
        y_pred.ravel(),
        zero_division=0
    )

    micro_recall = recall_score(
        y_true.ravel(),
        y_pred.ravel(),
        zero_division=0
    )

    micro_f1 = f1_score(
        y_true.ravel(),
        y_pred.ravel(),
        zero_division=0
    )

    # Print results
    print("\nPer-disease results")
    print("-------------------")

    for row in rows:

        print(
            f"{row['Disease']:22s} "
            f"AUC: {row['ROC_AUC']:.4f} "
            f"AP: {row['Average_Precision']:.4f} "
            f"F1: {row['F1_Score']:.4f}"
        )

    print("\nOverall results")
    print("---------------")

    print(f"Macro ROC-AUC: {macro_auc:.4f}")
    print(f"Micro ROC-AUC: {micro_auc:.4f}")
    print(f"Macro AP:      {macro_ap:.4f}")
    print(f"Micro AP:      {micro_ap:.4f}")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall:    {micro_recall:.4f}")
    print(f"Micro F1:        {micro_f1:.4f}")

    # Save summary
    summary_path = os.path.join(
        output_dir,
        "summary_metrics.txt"
    )

    with open(summary_path, "w") as file:

        file.write(f"Model: {args.model}\n")
        file.write(f"Threshold: {args.threshold}\n")
        file.write(f"Samples: {len(y_true)}\n")
        file.write(f"Macro ROC-AUC: {macro_auc:.6f}\n")
        file.write(f"Micro ROC-AUC: {micro_auc:.6f}\n")
        file.write(f"Macro AP: {macro_ap:.6f}\n")
        file.write(f"Micro AP: {micro_ap:.6f}\n")
        file.write(f"Micro Precision: {micro_precision:.6f}\n")
        file.write(f"Micro Recall: {micro_recall:.6f}\n")
        file.write(f"Micro F1: {micro_f1:.6f}\n")

    print("\nEvaluation completed!")
    print("Metrics:", metrics_path)
    print("Summary:", summary_path)
    print("Confusion matrices:", confusion_dir)


if __name__ == "__main__":
    evaluate()
