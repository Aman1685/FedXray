
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "NIH_ChestXray14"
SPLITS = ROOT / "dataset" / "splits"
RESULTS = ROOT / "results" / "graphs"

SPLITS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATASET / "Data_Entry_2017.csv"
TRAIN_VAL_PATH = DATASET / "train_val_list.txt"
TEST_PATH = DATASET / "test_list.txt"

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion",
    "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening",
    "Hernia"
]

print("\n========== FEDXRAY PREPROCESSING ==========\n")

# 1. Load annotation data
df = pd.read_csv(CSV_PATH)

df["Image Index"] = df["Image Index"].astype(str)
df["Patient ID"] = df["Patient ID"].astype(int)

# 2. Read official splits
def read_list(file):
    with open(file, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

train_val_names = read_list(TRAIN_VAL_PATH)
test_names = read_list(TEST_PATH)

# 3. Check patient leakage in official split
train_val_patients = set(
    df.loc[df["Image Index"].isin(train_val_names), "Patient ID"]
)

test_patients = set(
    df.loc[df["Image Index"].isin(test_names), "Patient ID"]
)

patient_overlap = train_val_patients & test_patients

print("Official train/validation patients:", len(train_val_patients))
print("Official test patients:", len(test_patients))
print("Patient overlap:", len(patient_overlap))

# 4. Create image path mapping
print("\nIndexing images...")

image_map = {
    p.name: str(p.resolve())
    for p in DATASET.rglob("*.png")
}

df["Image Path"] = df["Image Index"].map(image_map)

if df["Image Path"].isna().any():
    raise ValueError("Some images are missing.")

# 5. Encode disease labels
print("Encoding 14 disease labels...")

def encode_labels(label_string):
    labels = set(label_string.split("|"))
    return [int(disease in labels) for disease in DISEASES]

label_matrix = np.array(
    df["Finding Labels"].apply(encode_labels).tolist(),
    dtype=np.uint8
)

for i, disease in enumerate(DISEASES):
    df[disease] = label_matrix[:, i]

# 6. Create official train/validation and test dataframes
train_val_df = df[df["Image Index"].isin(train_val_names)].copy()
test_df = df[df["Image Index"].isin(test_names)].copy()

# 7. Patient-wise train-validation split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.10,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        train_val_df,
        groups=train_val_df["Patient ID"]
    )
)

train_df = train_val_df.iloc[train_idx].copy()
val_df = train_val_df.iloc[val_idx].copy()

# 8. Verify patient leakage
train_patients = set(train_df["Patient ID"])
val_patients = set(val_df["Patient ID"])

print("\n========== PATIENT LEAKAGE CHECK ==========")
print("Train/validation overlap:", len(train_patients & val_patients))
print("Train/test overlap:", len(train_patients & test_patients))
print("Validation/test overlap:", len(val_patients & test_patients))

if train_patients & val_patients:
    raise ValueError("Patient leakage between train and validation!")

if train_patients & test_patients or val_patients & test_patients:
    raise ValueError("Patient leakage with official test set!")

# 9. Save manifests
columns = [
    "Image Index",
    "Patient ID",
    "Finding Labels",
    "Image Path"
] + DISEASES

train_df[columns].to_csv(SPLITS / "train.csv", index=False)
val_df[columns].to_csv(SPLITS / "val.csv", index=False)
test_df[columns].to_csv(SPLITS / "test.csv", index=False)

print("\n========== SPLIT SUMMARY ==========")
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))

# 10. Disease distribution graph
counts = df[DISEASES].sum().sort_values(ascending=False)

plt.figure(figsize=(13, 6))
counts.plot(kind="bar")
plt.title("NIH ChestX-ray14 Disease Distribution")
plt.xlabel("Disease")
plt.ylabel("Number of Positive Images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(RESULTS / "disease_distribution.png", dpi=150)
plt.close()

# 11. Sample images with labels
from PIL import Image

sample = df.sample(6, random_state=42)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, (_, row) in zip(axes.flat, sample.iterrows()):
    with Image.open(row["Image Path"]) as img:
        ax.imshow(img.convert("L"), cmap="gray")

    ax.set_title(row["Finding Labels"][:45], fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig(RESULTS / "sample_xrays.png", dpi=150)
plt.close()

print("\n========== OUTPUT FILES ==========")
print("Train:", SPLITS / "train.csv")
print("Validation:", SPLITS / "val.csv")
print("Test:", SPLITS / "test.csv")
print("Disease graph:", RESULTS / "disease_distribution.png")
print("Sample images:", RESULTS / "sample_xrays.png")

print("\nPreprocessing completed successfully!")
