
import pandas as pd
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "NIH_ChestXray14"

CSV_FILE = DATASET / "Data_Entry_2017.csv"
TRAIN_FILE = DATASET / "train_val_list.txt"
TEST_FILE = DATASET / "test_list.txt"

print("\n========== FEDXRAY DATASET AUDIT ==========\n")

# 1. Check required files
for file in [CSV_FILE, TRAIN_FILE, TEST_FILE]:
    if not file.exists():
        raise FileNotFoundError(f"Missing file: {file}")
    print("FOUND:", file.name)

# 2. Load annotations
df = pd.read_csv(CSV_FILE)

print("\nTotal CSV records:", len(df))
print("CSV columns:", list(df.columns))

# 3. Find images recursively
print("\nScanning image directories...")

image_paths = list(DATASET.rglob("*.png"))

image_map = {p.name: p for p in image_paths}

print("PNG images found:", len(image_paths))

# 4. Check CSV-to-image matching
csv_names = set(df["Image Index"].astype(str))
image_names = set(image_map.keys())

matched = csv_names & image_names
missing = csv_names - image_names
extra = image_names - csv_names

print("\n========== IMAGE MATCHING ==========")
print("CSV image names:", len(csv_names))
print("Matched images:", len(matched))
print("Missing images:", len(missing))
print("Images without CSV records:", len(extra))

# 5. Read official split files
def read_split(path):
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

train_val = read_split(TRAIN_FILE)
test = read_split(TEST_FILE)

print("\n========== OFFICIAL SPLITS ==========")
print("Train + Validation list:", len(train_val))
print("Test list:", len(test))
print("Train/Test overlap:", len(train_val & test))

print("Missing train/validation images:",
      len(train_val - image_names))

print("Missing test images:",
      len(test - image_names))

print("Split names absent from CSV:",
      len((train_val | test) - csv_names))

print("CSV records absent from both lists:",
      len(csv_names - (train_val | test)))

# 6. Disease distribution
print("\n========== DISEASE DISTRIBUTION ==========")

disease_counts = Counter()

for labels in df["Finding Labels"].dropna():
    for disease in labels.split("|"):
        disease_counts[disease] += 1

for disease, count in disease_counts.most_common():
    percentage = count / len(df) * 100
    print(f"{disease:25s}: {count:7d} ({percentage:.2f}%)")

print("\nTotal distinct labels:", len(disease_counts))

# 7. Multi-label statistics
label_counts = df["Finding Labels"].fillna("").apply(
    lambda x: len(x.split("|")) if x else 0
)

print("\n========== MULTI-LABEL STATISTICS ==========")
print("Average labels per image:", round(label_counts.mean(), 3))
print("Maximum labels per image:", label_counts.max())
print("Images with No Finding:",
      (df["Finding Labels"] == "No Finding").sum())

# 8. Final audit
print("\n========== FINAL AUDIT ==========")

checks = {
    "CSV loaded": len(df) > 0,
    "Image matching complete": len(missing) == 0,
    "Train/test lists do not overlap": len(train_val & test) == 0,
    "All split images exist": (
        len(train_val - image_names) == 0
        and len(test - image_names) == 0
    ),
    "14 disease classes present": (
        len(set(disease_counts) - {"No Finding"}) == 14
    )
}

for name, status in checks.items():
    print(f"{name}: {'PASS' if status else 'CHECK'}")

print("\nDataset audit completed.")
