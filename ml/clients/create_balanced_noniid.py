
import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = r"D:\FedXray"

TRAIN_CSV = os.path.join(
    PROJECT_DIR, "dataset", "splits", "train.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR, "dataset", "splits", "clients_noniid_balanced"
)

NUM_CLIENTS = 5
DIRICHLET_ALPHA = 0.5
RANDOM_SEED = 42

DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]

PATIENT_COL = "Patient ID"


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

print("\nLoading training dataset...")

df = pd.read_csv(TRAIN_CSV)

required_columns = [PATIENT_COL] + DISEASES

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns in train.csv: {missing}")

df[PATIENT_COL] = df[PATIENT_COL].astype(str)

print("Total training images:", len(df))
print("Unique patients:", df[PATIENT_COL].nunique())


# ============================================================
# 2. GROUP IMAGES BY PATIENT
# ============================================================

print("\nGrouping images by patient...")

patient_records = []

for patient_id, group in df.groupby(PATIENT_COL):

    # Count disease labels across all images of this patient
    disease_counts = group[DISEASES].sum()

    if disease_counts.sum() == 0:
        dominant_label = "No Finding"
    else:
        dominant_label = disease_counts.idxmax()

    patient_records.append({
        "patient_id": patient_id,
        "image_count": len(group),
        "dominant_label": dominant_label
    })

patients = pd.DataFrame(patient_records)

print("Patient groups created:", len(patients))


# ============================================================
# 3. CREATE DIRICHLET LABEL PREFERENCES
# ============================================================

print("\nGenerating non-IID label preferences...")

rng = np.random.default_rng(RANDOM_SEED)

categories = patients["dominant_label"].unique()

preferences = {}

for category in categories:

    proportions = rng.dirichlet(
        np.repeat(DIRICHLET_ALPHA, NUM_CLIENTS)
    )

    preferences[category] = proportions

    print(f"{category}: {np.round(proportions, 3)}")



# ============================================================
# 4. ASSIGN PATIENTS WITH BALANCED IMAGE COUNTS
# ============================================================

print("\nAssigning patients to clients...")

target_images = len(df) / NUM_CLIENTS

# Hard maximum size: approximately 1% above target
upper_limit = int(np.ceil(target_images * 1.01))

client_patients = [[] for _ in range(NUM_CLIENTS)]
client_images = np.zeros(NUM_CLIENTS, dtype=int)

# Keep the Dirichlet preferences generated above.
# Process larger patient groups first to reduce packing problems.

patients = patients.sample(
    frac=1,
    random_state=RANDOM_SEED
)

patients = patients.sort_values(
    "image_count",
    ascending=False
)

# Randomize category processing order
categories = list(categories)
rng.shuffle(categories)

for category in categories:

    category_patients = patients[
        patients["dominant_label"] == category
    ].copy()

    category_patients = category_patients.sort_values(
        "image_count",
        ascending=False
    )

    category_preference = preferences[category]

    for _, patient in category_patients.iterrows():

        patient_id = patient["patient_id"]
        image_count = int(patient["image_count"])

        # Only clients that can fit this complete patient group
        eligible = [
            i for i in range(NUM_CLIENTS)
            if client_images[i] + image_count <= upper_limit
        ]

        if not eligible:
            # Try to fit into the client with the most remaining room.
            remaining = upper_limit - client_images
            eligible = [int(np.argmax(remaining))]

        # Balance client size while retaining label preferences.
        weights = np.array([
            category_preference[i] *
            np.exp(
                -4 * client_images[i] / target_images
            )
            for i in eligible
        ], dtype=float)

        if weights.sum() <= 0:
            weights = np.ones(len(eligible), dtype=float)

        weights /= weights.sum()

        chosen_client = int(
            rng.choice(eligible, p=weights)
        )

        client_patients[chosen_client].append(patient_id)
        client_images[chosen_client] += image_count


# ============================================================
# 4B. BALANCE CLIENT SIZES USING PATIENT-GROUP MOVES
# ============================================================

print("\nFine-tuning client sizes...")

# Move whole patients from the largest clients to the smallest.
# Never split a patient's images.

for iteration in range(10000):

    largest = int(np.argmax(client_images))
    smallest = int(np.argmin(client_images))

    difference = (
        client_images[largest] - client_images[smallest]
    )

    if difference <= target_images * 0.02:
        break

    best_patient = None
    best_difference = difference

    for patient_id in client_patients[largest]:

        patient_size = int(
            patients.loc[
                patients["patient_id"] == patient_id,
                "image_count"
            ].iloc[0]
        )

        new_largest = client_images[largest] - patient_size
        new_smallest = client_images[smallest] + patient_size

        # Only accept a move that improves the balance.
        new_difference = abs(new_largest - new_smallest)

        if new_difference < best_difference:
            best_difference = new_difference
            best_patient = patient_id

    if best_patient is None:
        break

    patient_size = int(
        patients.loc[
            patients["patient_id"] == best_patient,
            "image_count"
        ].iloc[0]
    )

    client_patients[largest].remove(best_patient)
    client_patients[smallest].append(best_patient)

    client_images[largest] -= patient_size
    client_images[smallest] += patient_size


print("\nFinal client image distribution:")

for i in range(NUM_CLIENTS):

    percentage = client_images[i] / len(df) * 100

    print(
        f"Client {i + 1}: "
        f"{client_images[i]} images "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 5. CHECK CLIENT BALANCE
# ============================================================

tolerance = 0.03

print("\nClient image distribution:")

for i in range(NUM_CLIENTS):

    percentage = (
        client_images[i] / len(df)
    ) * 100

    print(
        f"Client {i + 1}: "
        f"{client_images[i]} images "
        f"({percentage:.2f}%)"
    )

max_deviation = max(
    abs(client_images - target_images)
) / target_images

print(
    "\nMaximum deviation from target:",
    f"{max_deviation * 100:.2f}%"
)

if max_deviation > tolerance:
    raise RuntimeError(
        "Client sizes are outside the 3% tolerance. "
        "Try another RANDOM_SEED or adjust the assignment algorithm."
    )


# ============================================================
# 6. SAVE CLIENT CSV FILES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nSaving client CSV files...")

for i in range(NUM_CLIENTS):

    selected_patients = set(client_patients[i])

    client_df = df[
        df[PATIENT_COL].isin(selected_patients)
    ].copy()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"client_{i + 1}.csv"
    )

    client_df.to_csv(output_path, index=False)

    print(
        f"Client {i + 1} saved: "
        f"{len(client_df)} images"
    )



# ============================================================
# 7. VERIFY PATIENT LEAKAGE AND IMAGE COVERAGE
# ============================================================

print("\nVerifying partition...")

all_assigned_patients = set()
all_assigned_images = 0

for i in range(NUM_CLIENTS):

    path = os.path.join(
        OUTPUT_DIR,
        f"client_{i + 1}.csv"
    )

    client_df = pd.read_csv(path)

    all_assigned_images += len(client_df)

    # Unique patients within this client
    client_patient_ids = set(
        client_df[PATIENT_COL].astype(str).unique()
    )

    # Check whether any patient already belongs to another client
    overlap = all_assigned_patients.intersection(
        client_patient_ids
    )

    if overlap:
        raise AssertionError(
            f"Patient leakage in Client {i + 1}. "
            f"Overlapping patients: {list(overlap)[:10]}"
        )

    all_assigned_patients.update(client_patient_ids)

# Check that every training image was assigned exactly once
assert all_assigned_images == len(df), (
    f"Image count mismatch: "
    f"Expected {len(df)}, got {all_assigned_images}"
)

# Check that all original training patients are present
original_patients = set(
    df[PATIENT_COL].astype(str).unique()
)

assert all_assigned_patients == original_patients, (
    "Patient coverage mismatch!"
)

print("No patient leakage detected.")
print("All training patients assigned exactly once.")
print("All training images assigned exactly once.")


# ============================================================
# 8. PRINT DISEASE DISTRIBUTION
# ============================================================

print("\nDisease distribution by client:")

for i in range(NUM_CLIENTS):

    path = os.path.join(
        OUTPUT_DIR,
        f"client_{i + 1}.csv"
    )

    client_df = pd.read_csv(path)

    print(f"\nCLIENT {i + 1}")
    print("Images:", len(client_df))

    for disease in DISEASES:

        count = int(client_df[disease].sum())

        percentage = (
            count / len(client_df)
        ) * 100

        print(
            f"{disease:22s}: "
            f"{count:5d} ({percentage:6.2f}%)"
        )


print("\n========================================")
print("BALANCED NON-IID PARTITION COMPLETED")
print("========================================")
print("Output folder:", OUTPUT_DIR)
