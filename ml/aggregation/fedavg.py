
import torch


def fedavg_aggregate(client_states, client_sizes):
    """
    Federated Averaging (FedAvg).

    Args:
        client_states: List of client model state dictionaries.
        client_sizes: Number of training samples processed by each client.

    Returns:
        Aggregated global model state dictionary.
    """

    if not client_states:
        raise ValueError("No client models provided.")

    if len(client_states) != len(client_sizes):
        raise ValueError(
            "Number of client models and client sizes must match."
        )

    if any(size <= 0 for size in client_sizes):
        raise ValueError("Every client size must be positive.")

    total_samples = sum(client_sizes)

    # Calculate each client's aggregation weight
    weights = [
        size / total_samples
        for size in client_sizes
    ]

    # Validate that every client has the same model structure
    reference_keys = set(client_states[0].keys())

    for client_id, state in enumerate(client_states[1:], start=2):
        if set(state.keys()) != reference_keys:
            raise ValueError(
                f"Client {client_id} has different model parameters."
            )

    global_state = {}

    for key in client_states[0].keys():

        reference_tensor = client_states[0][key]

        for client_id, state in enumerate(client_states[1:], start=2):
            if state[key].shape != reference_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {key} in Client {client_id}."
                )

        # Weighted average for floating-point parameters and buffers
        if torch.is_floating_point(reference_tensor):

            aggregated = torch.zeros_like(
                reference_tensor,
                dtype=torch.float64,
                device="cpu"
            )

            for state, weight in zip(client_states, weights):

                aggregated += (
                    state[key].detach().cpu().to(torch.float64)
                    * weight
                )

            global_state[key] = aggregated.to(
                dtype=reference_tensor.dtype
            )

        else:
            # For integer buffers such as BatchNorm's
            # num_batches_tracked, retain the maximum client value.
            global_state[key] = torch.stack([
                state[key].detach().cpu()
                for state in client_states
            ]).max(dim=0).values

    print("\nFedAvg aggregation completed.")
    print(f"Total client samples: {total_samples}")

    for i, weight in enumerate(weights, start=1):
        print(f"Client {i} aggregation weight: {weight:.4f}")

    return global_state
