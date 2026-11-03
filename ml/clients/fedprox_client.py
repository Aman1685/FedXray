
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.nih_dataset import NIHChestXrayDataset


def train_fedprox_client(
    global_model,
    csv_file,
    client_id,
    device,
    local_epochs=1,
    batch_size=8,
    learning_rate=0.0001,
    max_batches=None,
    pos_weight=None,
    mu=0.01
):
    """
    Train one federated client using the FedProx algorithm.

    FedProx objective:

        Total Loss = BCE Loss + (mu / 2) * ||local_weights - global_weights||^2

    Returns:
        local_state_dict: Updated local model parameters
        num_samples: Number of processed training samples
        average_loss: Average classification loss
    """

    if mu <= 0:
        raise ValueError("FedProx mu must be greater than zero.")

    print(f"\nStarting FedProx training for Client {client_id}")
    print(f"Proximal coefficient (mu): {mu}")

    # --------------------------------------------------
    # 1. Load client's local dataset
    # --------------------------------------------------

    dataset = NIHChestXrayDataset(csv_file)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # --------------------------------------------------
    # 2. Preserve global model parameters
    # --------------------------------------------------

    global_params = [
        parameter.detach().clone()
        for parameter in global_model.parameters()
    ]

    # --------------------------------------------------
    # 3. Create local model from global model
    # --------------------------------------------------

    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    # --------------------------------------------------
    # 4. Classification loss
    # --------------------------------------------------

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=(
            pos_weight.to(device)
            if pos_weight is not None
            else None
        )
    )

    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=learning_rate
    )

    # --------------------------------------------------
    # 5. Mixed precision
    # --------------------------------------------------

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    total_classification_loss = 0.0
    total_samples = 0

    # --------------------------------------------------
    # 6. Local training
    # --------------------------------------------------

    for epoch in range(local_epochs):

        epoch_loss = 0.0
        epoch_samples = 0

        for batch_index, (images, labels) in enumerate(loader):

            if max_batches is not None and batch_index >= max_batches:
                break

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):

                # Local model predictions
                outputs = local_model(images)

                # Standard multi-label classification loss
                classification_loss = criterion(
                    outputs,
                    labels
                )

                # --------------------------------------
                # FedProx proximal regularization
                # --------------------------------------

                proximal_term = torch.zeros(
                    (),
                    device=device
                )

                for local_param, global_param in zip(
                    local_model.parameters(),
                    global_params
                ):

                    proximal_term += torch.sum(
                        (local_param - global_param) ** 2
                    )

                # Combined FedProx objective
                loss = classification_loss + (
                    (mu / 2.0) * proximal_term
                )

            # Backpropagation
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            # Track classification loss separately
            batch_samples = images.size(0)

            epoch_loss += (
                classification_loss.item() * batch_samples
            )

            epoch_samples += batch_samples

        if epoch_samples == 0:
            raise RuntimeError(
                f"Client {client_id} processed zero batches."
            )

        average_epoch_loss = epoch_loss / epoch_samples

        total_classification_loss += epoch_loss
        total_samples += epoch_samples

        print(
            f"Client {client_id} | "
            f"Epoch {epoch + 1}/{local_epochs} | "
            f"Classification Loss: {average_epoch_loss:.5f} | "
            f"Samples: {epoch_samples}"
        )

    # --------------------------------------------------
    # 7. Calculate average classification loss
    # --------------------------------------------------

    average_loss = (
        total_classification_loss / total_samples
    )

    # --------------------------------------------------
    # 8. Return updated local weights
    # --------------------------------------------------

    local_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in local_model.state_dict().items()
    }

    # --------------------------------------------------
    # 9. Free memory
    # --------------------------------------------------

    del local_model
    del optimizer
    del loader
    del global_params

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"FedProx Client {client_id} completed | "
        f"Average Classification Loss: {average_loss:.5f}"
    )

    return local_state_dict, total_samples, average_loss
