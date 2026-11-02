
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = r"D:\FedXray"

SPLIT_DIR = os.path.join(
    PROJECT_DIR, "dataset", "splits"
)

TRAIN_CSV = os.path.join(SPLIT_DIR, "train.csv")

PARTITIONS = {
    "IID": "clients_iid",
    "NonIID_Balanced": "clients_noniid_balanced",
    "NonIID_Quantity_Label": "clients_noniid"
}

OUTPUT_DIR = os.path.join(
    PROJECT_DIR, "results", "partition_analysis"
)

NUM_CLIENTS = 5
PATIENT_COL = "Patient ID"
IMAGE_COL = "Image Index"

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

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD ORIGINAL TRAINING DATA
# ============================================================

print("\nLoading original training data...")

train_df = pd.read_csv(TRAIN_CSV)

required_columns = (
    [PATIENT_COL, IMAGE_COL] + DISEASES
)

missing = [
    col for col in required_columns
    if col not in train_df.columns
]

if missing:
    raise ValueError(
        f"Missing columns in train.csv: {missing}"
    )

train_df[PATIENT_COL] = train_df[PATIENT_COL].astype(str)

original_patients = set(train_df[PATIENT_COL])

original_images = set(train_df[IMAGE_COL].astype(str))

print("Training images:", len(train_df))
print("Training patients:", len(original_patients))


# ============================================================
# 2. LOAD AND VALIDATE ALL CLIENT PARTITIONS
# ============================================================

print("\nLoading client partitions...")

all_data = {}

for partition_name, folder_name in PARTITIONS.items():

    print(f"\nChecking {partition_name}...")

    folder = os.path.join(SPLIT_DIR, folder_name)

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Partition folder not found: {folder}"
        )

    clients = {}

    assigned_patients = set()
    assigned_images = set()

    for client_number in range(1, NUM_CLIENTS + 1):

        filename = f"client_{client_number}.csv"

        path = os.path.join(folder, filename)

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

        client_df = pd.read_csv(path)

        missing = [
            col for col in required_columns
            if col not in client_df.columns
        ]

        if missing:
            raise ValueError(
                f"{filename} is missing columns: {missing}"
            )

        client_df[PATIENT_COL] = (
            client_df[PATIENT_COL].astype(str)
        )

        client_df[IMAGE_COL] = (
            client_df[IMAGE_COL].astype(str)
        )

        client_patients = set(
            client_df[PATIENT_COL].unique()
        )

        client_images = set(
            client_df[IMAGE_COL]
        )

        # Check patient overlap between clients
        overlap_patients = (
            assigned_patients.intersection(client_patients)
        )

        if overlap_patients:
            raise ValueError(
                f"Patient leakage in {partition_name}, "
                f"Client {client_number}: "
                f"{list(overlap_patients)[:5]}"
            )

        # Check duplicate images between clients
        overlap_images = (
            assigned_images.intersection(client_images)
        )

        if overlap_images:
            raise ValueError(
                f"Duplicate images in {partition_name}: "
                f"{list(overlap_images)[:5]}"
            )

        # Check that all client records belong to original train
        if not client_patients.issubset(original_patients):
            raise ValueError(
                f"Unknown patient found in {partition_name}"
            )

        if not client_images.issubset(original_images):
            raise ValueError(
                f"Unknown image found in {partition_name}"
            )

        assigned_patients.update(client_patients)
        assigned_images.update(client_images)

        clients[client_number] = client_df

        print(
            f"Client {client_number}: "
            f"{len(client_df)} images, "
            f"{len(client_patients)} patients"
        )

    # Check complete coverage
    if assigned_patients != original_patients:
        raise ValueError(
            f"Patient coverage mismatch in {partition_name}"
        )

    if assigned_images != original_images:
        raise ValueError(
            f"Image coverage mismatch in {partition_name}"
        )

    all_data[partition_name] = clients

    print(
        f"{partition_name}: all validation checks passed."
    )


# ============================================================
# 3. CLIENT SIZE AND PATIENT SUMMARY
# ============================================================

print("\nCalculating client size statistics...")

size_records = []
summary_records = []

for partition_name, clients in all_data.items():

    image_counts = []
    patient_counts = []

    for client_number, client_df in clients.items():

        images = len(client_df)
        patients = client_df[PATIENT_COL].nunique()

        image_counts.append(images)
        patient_counts.append(patients)

        size_records.append({
            "Partition": partition_name,
            "Client": f"Client {client_number}",
            "Images": images,
            "Patients": patients,
            "Image_Share_Percent": (
                images / len(train_df) * 100
            )
        })

    image_counts = np.array(image_counts, dtype=float)

    target = len(train_df) / NUM_CLIENTS

    max_deviation = (
        np.max(np.abs(image_counts - target))
        / target * 100
    )

    mean_images = np.mean(image_counts)

    std_images = np.std(image_counts)

    cv = (
        std_images / mean_images * 100
        if mean_images else 0
    )

    summary_records.append({
        "Partition": partition_name,
        "Total_Images": int(image_counts.sum()),
        "Total_Patients": len(original_patients),
        "Mean_Images_Per_Client": mean_images,
        "Min_Images": int(image_counts.min()),
        "Max_Images": int(image_counts.max()),
        "Image_Count_Range": int(
            image_counts.max() - image_counts.min()
        ),
        "Max_Deviation_Percent": max_deviation,
        "Image_Count_CV_Percent": cv
    })

size_df = pd.DataFrame(size_records)

summary_df = pd.DataFrame(summary_records)

size_df.to_csv(
    os.path.join(OUTPUT_DIR, "client_sizes.csv"),
    index=False
)

summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "partition_summary.csv"),
    index=False
)


# ============================================================
# 4. DISEASE PREVALENCE
# ============================================================

print("\nCalculating disease prevalence...")

prevalence_records = []

for partition_name, clients in all_data.items():

    for client_number, client_df in clients.items():

        images = len(client_df)

        for disease in DISEASES:

            positive_count = int(
                client_df[disease].sum()
            )

            prevalence = (
                positive_count / images * 100
                if images else 0
            )

            prevalence_records.append({
                "Partition": partition_name,
                "Client": f"Client {client_number}",
                "Disease": disease,
                "Positive_Images": positive_count,
                "Total_Images": images,
                "Prevalence_Percent": prevalence
            })

prevalence_df = pd.DataFrame(prevalence_records)

prevalence_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "disease_prevalence.csv"
    ),
    index=False
)

# Also save a wide-format table
prevalence_wide = prevalence_df.pivot_table(
    index=["Partition", "Client"],
    columns="Disease",
    values="Prevalence_Percent"
).reset_index()

prevalence_wide.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "disease_prevalence_wide.csv"
    ),
    index=False
)


# ============================================================
# 5. JENSEN-SHANNON DIVERGENCE
# ============================================================

def normalize_distribution(values):

    values = np.asarray(values, dtype=float)

    total = values.sum()

    if total == 0:
        return np.ones(len(values)) / len(values)

    return values / total


def kl_divergence(p, q):

    mask = p > 0

    return np.sum(
        p[mask] * np.log2(
            p[mask] / q[mask]
        )
    )


def js_divergence(p, q):

    p = normalize_distribution(p)
    q = normalize_distribution(q)

    midpoint = (p + q) / 2

    return (
        kl_divergence(p, midpoint)
        + kl_divergence(q, midpoint)
    ) / 2


print("\nCalculating Jensen-Shannon divergence...")

js_records = []
js_summary_records = []

for partition_name, clients in all_data.items():

    distributions = {}

    # Distribution of positive disease annotations.
    # Each image may contribute to multiple diseases.
    for client_number, client_df in clients.items():

        disease_counts = (
            client_df[DISEASES].sum().values
        )

        distributions[client_number] = (
            normalize_distribution(disease_counts)
        )

    pairwise_values = []

    for client_a, client_b in itertools.combinations(
        range(1, NUM_CLIENTS + 1), 2
    ):

        divergence = js_divergence(
            distributions[client_a],
            distributions[client_b]
        )

        pairwise_values.append(divergence)

        js_records.append({
            "Partition": partition_name,
            "Client_A": f"Client {client_a}",
            "Client_B": f"Client {client_b}",
            "JS_Divergence": divergence
        })

    # Compare each client with the pooled training distribution
    global_counts = train_df[DISEASES].sum().values

    global_distribution = normalize_distribution(
        global_counts
    )

    client_global_values = []

    for client_number in range(1, NUM_CLIENTS + 1):

        divergence = js_divergence(
            distributions[client_number],
            global_distribution
        )

        client_global_values.append(divergence)

        js_records.append({
            "Partition": partition_name,
            "Client_A": f"Client {client_number}",
            "Client_B": "Global",
            "JS_Divergence": divergence
        })

    js_summary_records.append({
        "Partition": partition_name,
        "Mean_Pairwise_JS": np.mean(pairwise_values),
        "Max_Pairwise_JS": np.max(pairwise_values),
        "Mean_Client_Global_JS": np.mean(
            client_global_values
        )
    })

js_df = pd.DataFrame(js_records)

js_summary_df = pd.DataFrame(js_summary_records)

js_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "js_divergence_details.csv"
    ),
    index=False
)

js_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "js_divergence_summary.csv"
    ),
    index=False
)


# ============================================================
# 6. GRAPH 1: CLIENT IMAGE COUNTS
# ============================================================

print("\nGenerating graphs...")

plt.figure(figsize=(12, 6))

for partition_name in PARTITIONS:

    subset = size_df[
        size_df["Partition"] == partition_name
    ]

    plt.plot(
        subset["Client"],
        subset["Images"],
        marker="o",
        label=partition_name
    )

plt.axhline(
    len(train_df) / NUM_CLIENTS,
    linestyle="--",
    label="Ideal equal size"
)

plt.title("Client Image Counts Across Partitions")
plt.xlabel("Client")
plt.ylabel("Number of Images")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "client_image_counts.png"),
    dpi=300
)

plt.close()


# ============================================================
# 7. GRAPH 2: DISEASE PREVALENCE HEATMAPS
# ============================================================

for partition_name in PARTITIONS:

    subset = prevalence_df[
        prevalence_df["Partition"] == partition_name
    ]

    matrix = subset.pivot(
        index="Client",
        columns="Disease",
        values="Prevalence_Percent"
    )

    matrix = matrix.reindex(
        index=[f"Client {i}" for i in range(1, 6)],
        columns=DISEASES
    )

    plt.figure(figsize=(15, 5))

    plt.imshow(
        matrix.values,
        aspect="auto"
    )

    plt.colorbar(
        label="Disease prevalence (%)"
    )

    plt.xticks(
        range(len(DISEASES)),
        DISEASES,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(NUM_CLIENTS),
        matrix.index
    )

    plt.title(
        f"Disease Prevalence - {partition_name}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"disease_heatmap_{partition_name}.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 8. GRAPH 3: JENSEN-SHANNON DIVERGENCE
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    js_summary_df["Partition"],
    js_summary_df["Mean_Pairwise_JS"]
)

plt.title(
    "Mean Pairwise Jensen-Shannon Divergence"
)

plt.xlabel("Partition")
plt.ylabel("JS Divergence (bits)")

plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "js_divergence_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 9. PRINT FINAL SUMMARY
# ============================================================

print("\n============================================")
print("PARTITION ANALYSIS SUMMARY")
print("============================================")

print("\nCLIENT SIZE SUMMARY")

print(
    summary_df.round(3).to_string(index=False)
)

print("\nJENSEN-SHANNON SUMMARY")

print(
    js_summary_df.round(4).to_string(index=False)
)

print("\n============================================")
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("============================================")

print("Reports saved to:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)
