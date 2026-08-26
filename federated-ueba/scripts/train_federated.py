from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, ServerApp
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from models.autoencoder import SecurityAutoencoder


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Flower ServerApp for federated security anomaly detection.

    Responsibilities:
    - Initialize the global anomaly-detection model
    - Configure FedAvg
    - Send training configuration to clients
    - Run federated rounds
    - Save the final global model
    """

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    input_dim = int(context.run_config.get("input-dim", 10))
    hidden_dim = int(context.run_config.get("hidden-dim", 32))
    latent_dim = int(context.run_config.get("latent-dim", 8))

    num_rounds = int(
        context.run_config.get("num-server-rounds", 5)
    )

    fraction_train = float(
        context.run_config.get("fraction-train", 1.0)
    )

    fraction_evaluate = float(
        context.run_config.get("fraction-evaluate", 1.0)
    )

    learning_rate = float(
        context.run_config.get("learning-rate", 0.001)
    )

    local_epochs = int(
        context.run_config.get("local-epochs", 2)
    )

    output_path = Path(
        context.run_config.get(
            "global-model-path",
            "artifacts/global_model.pt",
        )
    )

    # ---------------------------------------------------------
    # Initialize global model
    # ---------------------------------------------------------

    model = SecurityAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )

    # ArrayRecord is Flower's representation of the model
    # parameters sent between ServerApp and ClientApps.
    arrays = ArrayRecord(model.state_dict())

    # ---------------------------------------------------------
    # FedAvg strategy
    # ---------------------------------------------------------

    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=2,
        weighted_by_key="num-examples",
    )

    # ---------------------------------------------------------
    # Client training configuration
    # ---------------------------------------------------------

    train_config = ConfigRecord(
        {
            "learning-rate": learning_rate,
            "local-epochs": local_epochs,
        }
    )

    print("=" * 60)
    print("Federated Security Anomaly Detection")
    print("=" * 60)
    print(f"Rounds:             {num_rounds}")
    print(f"Training fraction:  {fraction_train}")
    print(f"Evaluation fraction:{fraction_evaluate}")
    print(f"Learning rate:      {learning_rate}")
    print(f"Local epochs:       {local_epochs}")
    print("=" * 60)

    # ---------------------------------------------------------
    # Start federated training
    # ---------------------------------------------------------

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        train_config=train_config,
    )

    # ---------------------------------------------------------
    # Save global model
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_state_dict = result.arrays.to_torch_state_dict()

    torch.save(
        final_state_dict,
        output_path,
    )

    print()
    print("=" * 60)
    print("Federated training complete")
    print("=" * 60)
    print(f"Global model saved to: {output_path}")