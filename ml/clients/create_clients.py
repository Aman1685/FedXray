
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "dataset" / "splits" / "train.csv"
OUTPUT = ROOT / "dataset" / "splits"

NUM_CLIENTS = 5
ALPHA = 0.5
SEED = 42

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion",
    "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening",
    "Hernia"
]

rng = np.random.default_rng(SEED)

df = pd.read_csv(TRAIN_PATH)

print("\n========== CLIENT PARTITIONING ==========")
print("Training images:", len(df))

# Create patient-level groups
patient_groups = {
    patient_id: group.copy()
    for patient_id, group in df.groupby("Patient ID")
}

patient_ids = list(patient_groups.keys())

print("Unique patients:", len(patient_ids))


def save_clients(assignments, experiment):

    output_dir = OUTPUT / experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    all_assigned = []

    print(f"\n========== {experiment.upper()} ==========")

    for i in range(NUM_CLIENTS):

        selected_patients = assignments[i]

        client_df = pd.concat(
            [patient_groups[p] for p in selected_patients],
            ignore_index=True
        )

        client_path = output_dir / f"client_{i+1}.csv"
        client_df.to_csv(client_path, index=False)

        all_assigned.extend(selected_patients)

        print(f"\nHospital {i+1}")
        print("Patients:", len(selected_patients))
        print("Images:", len(client_df))

        print("Disease distribution:")

        for disease in DISEASES:
            count = int(client_df[disease].sum())
            print(f"{disease:23s}: {count}")

    # Verify patient assignment
    assert len(all_assigned) == len(set(all_assigned))
    assert set(all_assigned) == set(patient_ids)

    print("\nAll patients assigned exactly once.")
    print("Patient leakage: 0")


# ==========================================
# EXPERIMENT A: IID PARTITION
# ==========================================

print("\nCreating IID clients...")

shuffled_patients = patient_ids.copy()
rng.shuffle(shuffled_patients)

iid_assignments = [[] for _ in range(NUM_CLIENTS)]
iid_sizes = [0] * NUM_CLIENTS

# Balance image counts while randomly distributing patients
for patient_id in sorted(
    shuffled_patients,
    key=lambda p: len(patient_groups[p]),
    reverse=True
):
    client = int(np.argmin(iid_sizes))

    iid_assignments[client].append(patient_id)
    iid_sizes[client] += len(patient_groups[patient_id])

save_clients(iid_assignments, "clients_iid")


# ==========================================
# EXPERIMENT B: NON-IID PARTITION
# ==========================================

print("\nCreating Non-IID clients...")

# Determine each patient's dominant category
patient_category = {}

for patient_id, group in patient_groups.items():

    disease_counts = group[DISEASES].sum(axis=0).to_numpy()

    if disease_counts.max() == 0:
        category = "No Finding"
    else:
        category = DISEASES[int(np.argmax(disease_counts))]

    patient_category[patient_id] = category


noniid_assignments = [[] for _ in range(NUM_CLIENTS)]

# Apply Dirichlet distribution within each category
categories = sorted(set(patient_category.values()))

for category in categories:

    category_patients = [
        p for p in patient_ids
        if patient_category[p] == category
    ]

    rng.shuffle(category_patients)

    proportions = rng.dirichlet(
        np.repeat(ALPHA, NUM_CLIENTS)
    )

    counts = rng.multinomial(
        len(category_patients),
        proportions
    )

    start = 0

    for client in range(NUM_CLIENTS):

        end = start + counts[client]

        noniid_assignments[client].extend(
            category_patients[start:end]
        )

        start = end

# Ensure no client is empty
empty_clients = [
    i for i in range(NUM_CLIENTS)
    if not noniid_assignments[i]
]

for empty_client in empty_clients:

    donor = max(
        range(NUM_CLIENTS),
        key=lambda i: len(noniid_assignments[i])
    )

    moved_patient = noniid_assignments[donor].pop()

    noniid_assignments[empty_client].append(moved_patient)

save_clients(noniid_assignments, "clients_noniid")


print("\n========== PARTITIONING COMPLETED ==========")
print("IID directory:", OUTPUT / "clients_iid")
print("Non-IID directory:", OUTPUT / "clients_noniid")
