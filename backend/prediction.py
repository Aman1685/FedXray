
import json
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model.custom_cnn import CustomCNN


# --------------------------------------------------
# PROJECT PATHS AND DEVICE
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
THRESHOLD_DIR = PROJECT_ROOT / "results" / "evaluation" / "thresholds"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# DISEASE LABELS
# --------------------------------------------------

DISEASE_LABELS = [
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


# --------------------------------------------------
# CHECKPOINT AND THRESHOLD REGISTRY
# --------------------------------------------------

MODEL_REGISTRY = {
    "centralized": {
        "name": "Centralized CNN",
        "configurations": {
            "standard": {
                "label": "Centralized",
                "checkpoint": "custom_cnn_centralized_best.pth",
                "thresholds": "centralized_thresholds.json"
            },
            "default": {
                "label": "Centralized",
                "checkpoint": "custom_cnn_centralized_best.pth",
                "thresholds": "centralized_thresholds.json"
            }
        }
    },

    "fedavg": {
        "name": "FedAvg",
        "configurations": {
            "iid": {
                "label": "IID",
                "checkpoint": "fedavg_custom_cnn_iid_seed42_best.pth",
                "thresholds": "fedavg_iid_thresholds.json"
            },
            "non-iid": {
                "label": "Non-IID",
                "checkpoint": "fedavg_custom_cnn_noniid_seed42_best.pth",
                "thresholds": "fedavg_noniid_thresholds.json"
            },
            "balanced-non-iid": {
                "label": "Balanced Non-IID",
                "checkpoint": "fedavg_custom_cnn_noniid_balanced_seed42_best.pth",
                "thresholds": "fedavg_balanced_noniid_thresholds.json"
            }
        }
    },

    "fedprox": {
        "name": "FedProx",
        "configurations": {
            "iid": {
                "label": "IID",
                "checkpoint": "fedprox_custom_cnn_clients_iid_seed42_best.pth",
                "thresholds": "fedprox_iid_thresholds.json"
            },
            "non-iid": {
                "label": "Non-IID",
                "checkpoint": "fedprox_custom_cnn_clients_noniid_seed42_best.pth",
                "thresholds": "fedprox_noniid_thresholds.json"
            },
            "balanced-non-iid": {
                "label": "Balanced Non-IID",
                "checkpoint": "fedprox_custom_cnn_clients_noniid_balanced_seed42_best.pth",
                "thresholds": "fedprox_balanced_noniid_thresholds.json"
            }
        }
    }
}


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# --------------------------------------------------
# SINGLE-MODEL CACHE
# --------------------------------------------------

_cached_model = None
_cached_model_key = None


# --------------------------------------------------
# CONFIGURATION NORMALIZATION
# --------------------------------------------------

def normalize_configuration(config_id):

    value = (config_id or "").strip().lower().replace("_", "-")

    aliases = {
        "default": "standard",
        "centralized": "standard",
        "iid-split": "iid",
        "non iid": "non-iid",
        "noniid": "non-iid",
        "balanced": "balanced-non-iid",
        "balanced-noniid": "balanced-non-iid",
        "balanced-non-iid-split": "balanced-non-iid"
    }

    return aliases.get(value, value)


def get_model_configuration(model_id, config_id):

    model_id = (model_id or "").strip().lower()
    config_id = normalize_configuration(config_id)

    if model_id not in MODEL_REGISTRY:
        raise ValueError(
            "Unknown model. Choose centralized, fedavg or fedprox."
        )

    configurations = MODEL_REGISTRY[model_id]["configurations"]

    if config_id not in configurations:
        available = ", ".join(configurations.keys())
        raise ValueError(
            f"Invalid configuration for {model_id}. "
            f"Available configurations: {available}"
        )

    return configurations[config_id]


# --------------------------------------------------
# THRESHOLD LOADING
# --------------------------------------------------

def load_thresholds(filename, threshold_mode="default"):

    threshold_mode = (threshold_mode or "default").strip().lower()

    if threshold_mode not in ("default", "tuned"):
        raise ValueError(
            "Invalid threshold mode. Choose 'default' or 'tuned'."
        )

    # Default mode: fixed threshold for all diseases.
    if threshold_mode == "default":
        return {
            disease: 0.5 for disease in DISEASE_LABELS
        }

    # Tuned mode: load disease-specific validation thresholds.
    threshold_path = THRESHOLD_DIR / filename

    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"Tuned threshold file not found: {threshold_path}"
        )

    with open(threshold_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    saved_diseases = data.get("diseases")
    saved_values = data.get("thresholds")

    if not isinstance(saved_diseases, list):
        raise ValueError(
            f"Invalid diseases list in {threshold_path.name}"
        )

    if not isinstance(saved_values, list):
        raise ValueError(
            f"Invalid thresholds list in {threshold_path.name}"
        )

    if len(saved_diseases) != len(saved_values):
        raise ValueError(
            f"Disease and threshold counts do not match in "
            f"{threshold_path.name}"
        )

    saved_mapping = {}

    for disease, value in zip(saved_diseases, saved_values):

        if disease not in DISEASE_LABELS:
            raise ValueError(
                f"Unknown disease '{disease}' in {threshold_path.name}"
            )

        value = float(value)

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Invalid threshold for {disease}: {value}"
            )

        saved_mapping[disease] = value

    missing_diseases = [
        disease for disease in DISEASE_LABELS
        if disease not in saved_mapping
    ]

    if missing_diseases:
        raise ValueError(
            f"Missing tuned thresholds for: {missing_diseases}"
        )

    return saved_mapping


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

def load_trained_model(model_id, config_id):

    global _cached_model, _cached_model_key

    model_id = (model_id or "").strip().lower()
    config_id = normalize_configuration(config_id)

    config = get_model_configuration(model_id, config_id)

    checkpoint_path = CHECKPOINT_DIR / config["checkpoint"]

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    cache_key = (model_id, config_id)

    if _cached_model is not None and _cached_model_key == cache_key:
        return _cached_model, config

    if _cached_model is not None:

        del _cached_model
        _cached_model = None
        _cached_model_key = None

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    model = CustomCNN(num_classes=14)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    if state_dict and all(
        key.startswith("module.") for key in state_dict.keys()
    ):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    _cached_model = model
    _cached_model_key = cache_key

    return model, config


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_xray(
    image_file,
    model_id,
    config_id,
    threshold_mode="default"
):

    model_id = (model_id or "").strip().lower()
    config_id = normalize_configuration(config_id)
    threshold_mode = (threshold_mode or "default").strip().lower()

    model, config = load_trained_model(model_id, config_id)

    try:
        image = Image.open(image_file).convert("RGB")

    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            "The uploaded file is not a valid image."
        ) from error

    image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)[0].cpu().tolist()

    thresholds = load_thresholds(
        config["thresholds"],
        threshold_mode
    )

    predictions = []

    for disease, probability in zip(DISEASE_LABELS, probabilities):

        threshold = thresholds[disease]

        predictions.append({
            "label": disease,
            "probability": round(float(probability), 6),
            "threshold": round(float(threshold), 6),
            "predicted": bool(probability >= threshold)
        })

    return {
        "success": True,
        "model_id": model_id,
        "configuration_id": config_id,
        "threshold_mode": threshold_mode,
        "model_name": MODEL_REGISTRY[model_id]["name"],
        "configuration_name": config["label"],
        "checkpoint": config["checkpoint"],
        "device": str(DEVICE),
        "predictions": predictions
    }
