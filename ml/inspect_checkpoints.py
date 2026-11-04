from pathlib import Path
import torch

CHECKPOINT_DIR = Path("results/checkpoints")

FILES = [
    "custom_cnn_centralized_best.pth",
    "fedavg_custom_cnn_iid_seed42_best.pth",
    "fedavg_custom_cnn_noniid_seed42_best.pth",
    "fedavg_custom_cnn_noniid_balanced_seed42_best.pth",
    "fedprox_custom_cnn_clients_iid_seed42_best.pth",
    "fedprox_custom_cnn_clients_noniid_seed42_best.pth",
    "fedprox_custom_cnn_clients_noniid_balanced_seed42_best.pth",
]

for filename in FILES:
    path = CHECKPOINT_DIR / filename

    print("\n" + "=" * 65)
    print("FILE:", filename)

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    print("Type:", type(checkpoint))

    if isinstance(checkpoint, dict):
        print("Keys:", list(checkpoint.keys()))

        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state = checkpoint
        else:
            print("No recognized model state dictionary.")
            continue

        print("Number of model tensors:", len(state))
        print("First five tensor shapes:")

        for key, value in list(state.items())[:5]:
            print(key, tuple(value.shape))

    else:
        print("Unexpected checkpoint format.")