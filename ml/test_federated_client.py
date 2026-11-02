
import torch

from model.custom_cnn import CustomCNN
from ml.clients.federated_client import train_client


def main():

    # Select GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Initialize a fresh Custom CNN
    global_model = CustomCNN(num_classes=14)

    # Keep an original copy for comparison
    original_weights = {
        key: value.detach().cpu().clone()
        for key, value in global_model.state_dict().items()
    }

    # Train only Client 1 for this test
    client_csv = "dataset/splits/clients_iid/client_1.csv"

    updated_weights, num_samples, average_loss = train_client(
        global_model=global_model,
        csv_file=client_csv,
        client_id=1,
        device=device,
        local_epochs=1,
        batch_size=8,
        learning_rate=0.0001,
        max_batches=20
    )

    # Verify training results
    assert num_samples > 0, "No samples were processed."

    assert torch.isfinite(torch.tensor(average_loss)), (
        "Training loss is not finite."
    )

    # Check whether model weights changed
    changed_parameters = []

    for key in original_weights:

        if not torch.equal(
            original_weights[key],
            updated_weights[key]
        ):
            changed_parameters.append(key)

    print("\n========== CLIENT TEST RESULTS ==========")
    print("Client ID:", 1)
    print("Samples processed:", num_samples)
    print("Average training loss:", round(average_loss, 5))
    print("Changed model tensors:", len(changed_parameters))
    print("Total model tensors:", len(original_weights))

    if not changed_parameters:
        raise RuntimeError("Model weights did not change.")

    print("\nSUCCESS: Local client training is working!")


if __name__ == "__main__":
    main()
