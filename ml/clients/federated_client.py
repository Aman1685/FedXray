
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.nih_dataset import NIHChestXrayDataset


def train_client(
    global_model,
    csv_file,
    client_id,
    device,
    local_epochs=1,
    batch_size=8,
    learning_rate=0.0001,
    max_batches=None,
    pos_weight=None
):
    """
    Train one federated client using its local dataset.

    Returns:
        local_state_dict: Updated model parameters
        num_samples: Number of local training samples
        average_loss: Average local training loss
    """

    print(f"\nStarting training for Client {client_id}")

    # Load local dataset
    dataset = NIHChestXrayDataset(csv_file)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Create a local copy of the global model
    local_model = copy.deepcopy(global_model)
    local_model = local_model.to(device)
    local_model.train()

    # Multi-label classification loss
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

    # Mixed precision for GPU memory efficiency
    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    total_loss = 0.0
    total_samples = 0
    total_batches = 0

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

                outputs = local_model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_samples = images.size(0)

            epoch_loss += loss.item() * batch_samples
            epoch_samples += batch_samples

        if epoch_samples == 0:
            raise RuntimeError(
                f"Client {client_id} processed zero batches."
            )

        average_epoch_loss = epoch_loss / epoch_samples

        total_loss += epoch_loss
        total_samples += epoch_samples
        total_batches += 1

        print(
            f"Client {client_id} | "
            f"Epoch {epoch + 1}/{local_epochs} | "
            f"Loss: {average_epoch_loss:.5f} | "
            f"Samples: {epoch_samples}"
        )

    average_loss = total_loss / total_samples

    # Return model weights on CPU to save GPU memory
    local_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in local_model.state_dict().items()
    }

    del local_model
    del optimizer
    del loader

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"Client {client_id} completed | "
        f"Average Loss: {average_loss:.5f}"
    )

    return local_state_dict, total_samples, average_loss
