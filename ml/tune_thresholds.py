
import os
import json
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, f1_score

from dataset.nih_dataset import NIHChestXrayDataset, DISEASE_LABELS
from model.model_factory import get_model


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VAL_CSV = os.path.join(ROOT, "dataset", "splits", "val.csv")
CHECKPOINT_DIR = os.path.join(ROOT, "results", "checkpoints")
OUTPUT_DIR = os.path.join(ROOT, "results", "evaluation", "thresholds")

BATCH_SIZE = 8
DEFAULT_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "centralized": "custom_cnn_centralized_best.pth",
    "fedavg_iid": "fedavg_custom_cnn_iid_seed42_best.pth",
    "fedavg_noniid": "fedavg_custom_cnn_noniid_seed42_best.pth",
    "fedavg_balanced_noniid": "fedavg_custom_cnn_noniid_balanced_seed42_best.pth",
    "fedprox_iid": "fedprox_custom_cnn_clients_iid_seed42_best.pth",
    "fedprox_noniid": "fedprox_custom_cnn_clients_noniid_seed42_best.pth",
    "fedprox_balanced_noniid": "fedprox_custom_cnn_clients_noniid_balanced_seed42_best.pth",
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


def get_validation_predictions(model, loader):
    all_probabilities = []
    all_labels = []

    use_amp = DEVICE.type == "cuda"

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=use_amp
            ):
                logits = model(images)

            probabilities = torch.sigmoid(logits)

            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(labels.numpy())

    return (
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_probabilities, axis=0)
    )


def find_best_thresholds(y_true, probabilities):
    thresholds = []
    results = []

    for i, disease in enumerate(DISEASE_LABELS):
        actual = y_true[:, i].astype(int)
        scores = probabilities[:, i]

        positive_count = int(actual.sum())

        if positive_count == 0:
            best_threshold = DEFAULT_THRESHOLD
            best_f1 = 0.0
            print(f"{disease}: no positive validation samples; using 0.50")

        else:
            precision, recall, candidate_thresholds = (
                precision_recall_curve(actual, scores)
            )

            # The final precision/recall point has no corresponding threshold.
            precision = precision[:-1]
            recall = recall[:-1]

            denominator = precision + recall

            f1_values = np.divide(
                2 * precision * recall,
                denominator,
                out=np.zeros_like(denominator),
                where=denominator != 0
            )

            best_index = int(np.argmax(f1_values))
            best_threshold = float(candidate_thresholds[best_index])
            best_f1 = float(f1_values[best_index])

        thresholds.append(best_threshold)

        predictions = (scores >= best_threshold).astype(int)

        results.append({
            "Disease": disease,
            "Positive_Count": positive_count,
            "Default_Threshold": DEFAULT_THRESHOLD,
            "Default_F1": f1_score(
                actual, (scores >= DEFAULT_THRESHOLD).astype(int),
                zero_division=0
            ),
            "Optimized_Threshold": best_threshold,
            "Optimized_F1_Validation": best_f1,
            "Tuned_Predicted_Positive_Count": int(predictions.sum())
        })

        print(
            f"{disease:22s} | "
            f"threshold={best_threshold:.4f} | "
            f"validation F1={best_f1:.4f}"
        )

    return thresholds, results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("VALIDATION-ONLY PER-DISEASE THRESHOLD TUNING")
    print("=" * 65)
    print("Device:", DEVICE)

    dataset = NIHChestXrayDataset(
        csv_file=VAL_CSV,
        image_size=224
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    print("Validation images:", len(dataset))
    print("Diseases:", len(DISEASE_LABELS))

    summary_rows = []

    for model_name, checkpoint_name in MODELS.items():
        print("\n" + "=" * 65)
        print("Processing:", model_name)
        print("=" * 65)

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR, checkpoint_name
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)

        model = load_model(checkpoint_path)

        y_true, probabilities = get_validation_predictions(
            model, loader
        )

        thresholds, results = find_best_thresholds(
            y_true, probabilities
        )

        threshold_path = os.path.join(
            OUTPUT_DIR, f"{model_name}_thresholds.json"
        )

        with open(threshold_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "model": model_name,
                    "source": "validation_only",
                    "default_threshold": DEFAULT_THRESHOLD,
                    "diseases": DISEASE_LABELS,
                    "thresholds": thresholds
                },
                file,
                indent=4
            )

        metrics_path = os.path.join(
            OUTPUT_DIR, f"{model_name}_threshold_metrics.csv"
        )

        pd.DataFrame(results).to_csv(metrics_path, index=False)

        for row in results:
            summary_rows.append({
                "Model": model_name,
                **row
            })

        print("Saved thresholds:", threshold_path)
        print("Saved validation metrics:", metrics_path)

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    summary_path = os.path.join(
        OUTPUT_DIR, "all_models_threshold_summary.csv"
    )

    pd.DataFrame(summary_rows).to_csv(
        summary_path, index=False
    )

    print("\n" + "=" * 65)
    print("THRESHOLD TUNING COMPLETED")
    print("=" * 65)
    print("Summary:", summary_path)
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
