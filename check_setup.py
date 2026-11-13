"""
check_setup.py - run this BEFORE building the backend:   python check_setup.py
It only READS files. It changes nothing.
"""

import json
import sys
from pathlib import Path

# The folder this file lives in is the project folder
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))   # lets "from model.model_factory import ..." work

CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
THRESHOLD_DIR = PROJECT_ROOT / "results" / "evaluation" / "thresholds"

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

# Copied from your teammate's Test_prediction.py:  name -> (checkpoint file, threshold file)
EXPECTED_FILES = {
    "Centralized / Default": ("custom_cnn_centralized_best.pth", "centralized_thresholds.json"),
    "FedAvg / IID": ("fedavg_custom_cnn_iid_seed42_best.pth", "fedavg_iid_thresholds.json"),
    "FedAvg / Non-IID": ("fedavg_custom_cnn_noniid_seed42_best.pth", "fedavg_noniid_thresholds.json"),
    "FedAvg / Balanced Non-IID": ("fedavg_custom_cnn_noniid_balanced_seed42_best.pth", "fedavg_balanced_noniid_thresholds.json"),
    "FedProx / IID": ("fedprox_custom_cnn_clients_iid_seed42_best.pth", "fedprox_iid_thresholds.json"),
    "FedProx / Non-IID": ("fedprox_custom_cnn_clients_noniid_seed42_best.pth", "fedprox_noniid_thresholds.json"),
    "FedProx / Balanced Non-IID": ("fedprox_custom_cnn_clients_noniid_balanced_seed42_best.pth", "fedprox_balanced_noniid_thresholds.json"),
}


def status(ok):
    return "OK     " if ok else "MISSING"


print("PROJECT FOLDER:", PROJECT_ROOT)

# 1. Folders ------------------------------------------------------------
print("\n1. Folders")
for folder in [PROJECT_ROOT / "model", CHECKPOINT_DIR, THRESHOLD_DIR]:
    print("  ", status(folder.is_dir()), folder.relative_to(PROJECT_ROOT))

# 2. Files your teammate's code expects -----------------------------------
print("\n2. Expected files")
for name, (checkpoint_name, threshold_name) in EXPECTED_FILES.items():
    print("  ", name)
    print("      ", status((CHECKPOINT_DIR / checkpoint_name).is_file()), checkpoint_name)
    print("      ", status((THRESHOLD_DIR / threshold_name).is_file()), threshold_name)

# 3. Files that exist but are not in the list (possible spelling differences) ---
print("\n3. Other files found in those folders")
expected_names = {file_name for pair in EXPECTED_FILES.values() for file_name in pair}
for folder, pattern in [(CHECKPOINT_DIR, "*.pth"), (THRESHOLD_DIR, "*.json")]:
    if folder.is_dir():
        for path in sorted(folder.glob(pattern)):
            if path.name not in expected_names:
                print("  ", path.relative_to(PROJECT_ROOT))

# 4. Python packages -------------------------------------------------------
print("\n4. Python packages")
print("   Python:", sys.version.split()[0])
try:
    import torch
    import torchvision
    import PIL
    print("   torch:", torch.__version__, "| torchvision:", torchvision.__version__,
          "| Pillow:", PIL.__version__, "| CUDA available:", torch.cuda.is_available())
except Exception as error:
    print("   PROBLEM importing torch / torchvision / Pillow:", error)
    sys.exit(1)

# 5. Can we build the model from your teammate's code? -------------------------
print("\n5. Model code")
try:
    from model.model_factory import get_model
    model = get_model("custom_cnn")
    print("   OK: custom_cnn built with", sum(p.numel() for p in model.parameters()), "parameters")
except Exception as error:
    print("   PROBLEM:", type(error).__name__, error)
    sys.exit(1)

# 6. Does every checkpoint load into that model? ---------------------------------
print("\n6. Loading each checkpoint into the model")
for name, (checkpoint_name, _) in EXPECTED_FILES.items():
    path = CHECKPOINT_DIR / checkpoint_name
    if not path.is_file():
        print("  ", name, "-> skipped (file missing)")
        continue
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict, found = checkpoint["model_state_dict"], "model_state_dict"
        elif "state_dict" in checkpoint:
            state_dict, found = checkpoint["state_dict"], "state_dict"
        else:
            state_dict, found = checkpoint, "plain state dict"
        get_model("custom_cnn").load_state_dict(state_dict)
        print("  ", name, "-> OK (", found, ")")
    except Exception as error:
        print("  ", name, "-> FAILED:", type(error).__name__, str(error)[:300])

# 7. Are the threshold files in the format the code expects? ------------------------
print("\n7. Threshold files")
for name, (_, threshold_name) in EXPECTED_FILES.items():
    path = THRESHOLD_DIR / threshold_name
    if not path.is_file():
        print("  ", name, "-> skipped (file missing)")
        continue
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        values = data.get("thresholds", {})
        missing = [label for label in DISEASE_LABELS if label not in values]
        if missing:
            print("  ", name, "-> PROBLEM, no threshold for:", ", ".join(missing))
        else:
            numbers = [float(values[label]) for label in DISEASE_LABELS]
            print("  ", name, "-> OK (14 thresholds, from", round(min(numbers), 3), "to", round(max(numbers), 3), ")")
    except Exception as error:
        print("  ", name, "-> FAILED:", type(error).__name__, str(error)[:300])

print("\nDone. Copy everything printed above and paste it to me.")