
import os
import json
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

from dataset.nih_dataset import NIHChestXrayDataset, DISEASE_LABELS
from model.model_factory import get_model


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_CSV = os.path.join(ROOT, "dataset", "splits", "test.csv")

CHECKPOINT_DIR = os.path.join(ROOT, "results", "checkpoints")
THRESHOLD_DIR = os.path.join(
    ROOT, "results", "evaluation", "thresholds"
)

OUTPUT_DIR = os.path.join(
    ROOT, "results", "evaluation", "official_test_tuned"
)

BATCH_SIZE = 8
DEFAULT_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "centralized": {
        "checkpoint": "custom_cnn_centralized_best.pth",
        "thresholds": "centralized_thresholds.json"
    },
    "fedavg_iid": {
        "checkpoint": "fedavg_custom_cnn_iid_seed42_best.pth",
        "thresholds": "fedavg_iid_thresholds.json"
    },
    "fedavg_noniid": {
        "checkpoint": "fedavg_custom_cnn_noniid_seed42_best.pth",
        "thresholds": "fedavg_noniid_thresholds.json"
    },
    "fedavg_balanced_noniid": {
        "checkpoint": "fedavg_custom_cnn_noniid_balanced_seed42_best.pth",
        "thresholds": "fedavg_balanced_noniid_thresholds.json"
    },
    "fedprox_iid": {
        "checkpoint": "fedprox_custom_cnn_clients_iid_seed42_best.pth",
        "thresholds": "fedprox_iid_thresholds.json"
    },
    "fedprox_noniid": {
        "checkpoint": "fedprox_custom_cnn_clients_noniid_seed42_best.pth",
        "thresholds": "fedprox_noniid_thresholds.json"
    },
    "fedprox_balanced_noniid": {
        "checkpoint": "fedprox_custom_cnn_clients_noniid_balanced_seed42_best.pth",
        "thresholds": "fedprox_balanced_noniid_thresholds.json"
    }
}


def load_model(checkpoint_path):
    model = get_model("custom_cnn").to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def load_thresholds(threshold_path):
    with open(threshold_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if data.get("source") != "validation_only":
        raise ValueError(
            f"Thresholds are not marked validation_only: {threshold_path}"
        )

    if data["diseases"] != DISEASE_LABELS:
        raise ValueError(
            f"Disease order mismatch in {threshold_path}"
        )

    thresholds = np.asarray(data["thresholds"], dtype=np.float32)

    if thresholds.shape != (len(DISEASE_LABELS),):
        raise ValueError(
            f"Expected {len(DISEASE_LABELS)} thresholds, "
            f"got {thresholds.shape}"
        )

    if not np.all(np.isfinite(thresholds)):
        raise ValueError("Non-finite threshold detected.")

    if np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("Thresholds must be between 0 and 1.")

    return thresholds


def get_test_predictions(model, loader):
    all_probabilities = []
    all_labels = []

    use_amp = DEVICE.type == "cuda"

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(DEVICE)

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=use_amp
            ):
                logits = model(images)

            probabilities = torch.sigmoid(logits)

            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(labels.numpy())

            if batch_index % 500 == 0:
                print(
                    f"  Processed batches: {batch_index}/{len(loader)}"
                )

    return (
        np.concatenate(all_labels, axis=0).astype(int),
        np.concatenate(all_probabilities, axis=0)
    )


def calculate_metrics(y_true, probabilities, thresholds):
    predictions = (
        probabilities >= thresholds.reshape(1, -1)
    ).astype(int)

    metrics = {}

    metrics["Exact_Match_Accuracy"] = accuracy_score(
        y_true, predictions
    )

    metrics["Per_Label_Accuracy"] = accuracy_score(
        y_true.ravel(), predictions.ravel()
    )

    metrics["Hamming_Loss"] = hamming_loss(
        y_true, predictions
    )

    metrics["Macro_Precision"] = precision_score(
        y_true, predictions, average="macro", zero_division=0
    )

    metrics["Micro_Precision"] = precision_score(
        y_true, predictions, average="micro", zero_division=0
    )

    metrics["Macro_Recall"] = recall_score(
        y_true, predictions, average="macro", zero_division=0
    )

    metrics["Micro_Recall"] = recall_score(
        y_true, predictions, average="micro", zero_division=0
    )

    metrics["Macro_F1"] = f1_score(
        y_true, predictions, average="macro", zero_division=0
    )

    metrics["Micro_F1"] = f1_score(
        y_true, predictions, average="micro", zero_division=0
    )

    try:
        metrics["Macro_ROC_AUC"] = roc_auc_score(
            y_true, probabilities, average="macro"
        )
    except ValueError:
        metrics["Macro_ROC_AUC"] = np.nan

    try:
        metrics["Micro_ROC_AUC"] = roc_auc_score(
            y_true, probabilities, average="micro"
        )
    except ValueError:
        metrics["Micro_ROC_AUC"] = np.nan

    try:
        metrics["Macro_AP"] = average_precision_score(
            y_true, probabilities, average="macro"
        )
    except ValueError:
        metrics["Macro_AP"] = np.nan

    try:
        metrics["Micro_AP"] = average_precision_score(
            y_true, probabilities, average="micro"
        )
    except ValueError:
        metrics["Micro_AP"] = np.nan

    return predictions, metrics


def calculate_disease_metrics(
    model_name,
    y_true,
    probabilities,
    predictions,
    thresholds
):
    rows = []

    for i, disease in enumerate(DISEASE_LABELS):
        actual = y_true[:, i]
        predicted = predictions[:, i]
        scores = probabilities[:, i]

        tn = int(((actual == 0) & (predicted == 0)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())
        tp = int(((actual == 1) & (predicted == 1)).sum())

        if len(np.unique(actual)) == 2:
            auc = roc_auc_score(actual, scores)
        else:
            auc = np.nan

        ap = average_precision_score(actual, scores)

        rows.append({
            "Model": model_name,
            "Disease": disease,
            "Positive_Count": int(actual.sum()),
            "Prevalence": float(actual.mean()),
            "Threshold": float(thresholds[i]),
            "Accuracy": accuracy_score(actual, predicted),
            "ROC_AUC": auc,
            "Average_Precision": ap,
            "Precision": precision_score(
                actual, predicted, zero_division=0
            ),
            "Recall": recall_score(
                actual, predicted, zero_division=0
            ),
            "F1_Score": f1_score(
                actual, predicted, zero_division=0
            ),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("OFFICIAL TEST EVALUATION: DEFAULT VS TUNED THRESHOLDS")
    print("=" * 70)
    print("Device:", DEVICE)

    dataset = NIHChestXrayDataset(
        csv_file=TEST_CSV,
        image_size=224
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    print("Official test images:", len(dataset))
    print("Number of diseases:", len(DISEASE_LABELS))

    summary_rows = []
    disease_rows = []

    for model_name, paths in MODELS.items():
        print("\n" + "=" * 70)
        print("Evaluating:", model_name)
        print("=" * 70)

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR, paths["checkpoint"]
        )

        threshold_path = os.path.join(
            THRESHOLD_DIR, paths["thresholds"]
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)

        if not os.path.exists(threshold_path):
            raise FileNotFoundError(threshold_path)

        thresholds = load_thresholds(threshold_path)

        model = load_model(checkpoint_path)

        y_true, probabilities = get_test_predictions(
            model, loader
        )

        # Evaluation with the original 0.5 threshold.
        default_thresholds = np.full(
            len(DISEASE_LABELS),
            DEFAULT_THRESHOLD,
            dtype=np.float32
        )

        default_predictions, default_metrics = calculate_metrics(
            y_true, probabilities, default_thresholds
        )

        # Evaluation with thresholds selected on validation data.
        tuned_predictions, tuned_metrics = calculate_metrics(
            y_true, probabilities, thresholds
        )

        for setting, metrics in [
            ("default_0.5", default_metrics),
            ("validation_tuned", tuned_metrics)
        ]:
            summary_rows.append({
                "Model": model_name,
                "Threshold_Setting": setting,
                **metrics
            })

        disease_rows.extend(
            calculate_disease_metrics(
                model_name,
                y_true,
                probabilities,
                default_predictions,
                default_thresholds
            )
        )

        tuned_disease_rows = calculate_disease_metrics(
            model_name,
            y_true,
            probabilities,
            tuned_predictions,
            thresholds
        )

        for row in tuned_disease_rows:
            row["Threshold_Setting"] = "validation_tuned"

        default_disease_rows = [
            row for row in disease_rows
            if row["Model"] == model_name
        ]

        for row in default_disease_rows:
            row["Threshold_Setting"] = "default_0.5"

        disease_rows.extend(tuned_disease_rows)

        print("\nDefault threshold metrics:")
        for key, value in default_metrics.items():
            print(f"  {key}: {value:.5f}")

        print("\nValidation-tuned threshold metrics:")
        for key, value in tuned_metrics.items():
            print(f"  {key}: {value:.5f}")

        print(
            "\nMicro F1 change:",
            f"{default_metrics['Micro_F1']:.5f}",
            "->",
            f"{tuned_metrics['Micro_F1']:.5f}"
        )

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    disease_df = pd.DataFrame(disease_rows)

    summary_path = os.path.join(
        OUTPUT_DIR, "all_models_default_vs_tuned_summary.csv"
    )

    disease_path = os.path.join(
        OUTPUT_DIR, "all_models_default_vs_tuned_disease_metrics.csv"
    )

    summary_df.to_csv(summary_path, index=False)
    disease_df.to_csv(disease_path, index=False)

    print("\n" + "=" * 70)
    print("FINAL OFFICIAL TEST EVALUATION COMPLETED")
    print("=" * 70)

    print("Summary:", summary_path)
    print("Disease metrics:", disease_path)

    print("\nMicro F1 comparison:")
    print(
        summary_df.pivot(
            index="Model",
            columns="Threshold_Setting",
            values="Micro_F1"
        ).to_string()
    )


if __name__ == "__main__":
    main()
