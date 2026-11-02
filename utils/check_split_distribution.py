
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "dataset" / "splits"
OUTPUT = ROOT / "results" / "graphs"

OUTPUT.mkdir(parents=True, exist_ok=True)

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion",
    "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening",
    "Hernia"
]

datasets = {
    "Train": pd.read_csv(SPLITS / "train.csv"),
    "Validation": pd.read_csv(SPLITS / "val.csv"),
    "Test": pd.read_csv(SPLITS / "test.csv")
}

print("\n========== SPLIT DISTRIBUTION AUDIT ==========")

summary = {}

for name, df in datasets.items():

    counts = df[DISEASES].sum()
    percentages = counts / len(df) * 100

    no_finding = (
        df["Finding Labels"] == "No Finding"
    ).sum()

    avg_labels = df[DISEASES].sum(axis=1).mean()

    summary[name] = counts

    print(f"\n{name.upper()}")
    print("Total images:", len(df))
    print("No Finding:", no_finding)
    print("Average positive labels:", round(avg_labels, 3))

    for disease in DISEASES:
        print(
            f"{disease:23s}: "
            f"{counts[disease]:6d} "
            f"({percentages[disease]:.2f}%)"
        )

# Compare positive disease counts
comparison = pd.DataFrame(summary)

print("\n========== DISEASE COMPARISON ==========")
print(comparison.to_string())

# Plot disease distribution across splits
comparison.plot(kind="bar", figsize=(15, 7))

plt.title("Disease Distribution Across Dataset Splits")
plt.xlabel("Disease")
plt.ylabel("Positive Image Count")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Dataset")
plt.tight_layout()

plt.savefig(
    OUTPUT / "split_disease_comparison.png",
    dpi=150
)

plt.close()

print("\nGraph saved successfully!")
print(OUTPUT / "split_disease_comparison.png")
