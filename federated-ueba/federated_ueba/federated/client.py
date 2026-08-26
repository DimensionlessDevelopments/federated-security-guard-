"""Flower ClientApp for federated behavioural anomaly detection.

Each client represents a station that trains the autoencoder on its
local normal activity data and evaluates using a mix of normal + attack events.

Uses the Flower message API (matching the ServerApp's `strategy.start`):
the server sends ArrayRecord/ConfigRecord messages, the client replies
with updated arrays and scalar metrics -- raw event data never leaves.
"""

from __future__ import annotations

import numpy as np
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from federated_ueba.data.generator import (
    NUM_FEATURES,
    STATION_NAMES,
    generate_normal_events,
)
from federated_ueba.data.preprocessing import normalize_features
from federated_ueba.detection.scoring import reconstruction_error
from federated_ueba.models import (
    SecurityAutoencoder,
    get_parameters,
    set_parameters,
)
from federated_ueba.simulation.attack_generator import generate_mixed_dataset

BATCH_SIZE = 32
LOCAL_EPOCHS = 10
LEARNING_RATE = 0.001

# Keys of the MetricRecord in this client's replies. FedAvg requires every
# reply in a round to carry IDENTICAL MetricRecord keys, so any other node
# type in the run (e.g. Station B) must mirror these exact sets.
TRAIN_METRIC_KEYS = ("num-examples",)
EVAL_METRIC_KEYS = (
    "eval_loss",
    "num-examples",
    "accuracy",
    "precision",
    "recall",
    "fpr",
    "threshold",
)


class StationClient:
    """Local trainer for one station. Transport-agnostic: works with plain
    ndarray lists so it can run with or without the Flower runtime."""

    def __init__(
        self,
        station: str,
        seed: int,
        hidden_dim: int = 32,
        latent_dim: int = 8,
    ) -> None:
        # input_dim is fixed by the feature schema; hidden/latent dims must
        # match the server's run config so parameter shapes line up.
        self.model = SecurityAutoencoder(
            input_dim=NUM_FEATURES,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )
        self.station = station

        normal_data = generate_normal_events(station, n_samples=500, seed=seed)
        self.features = normal_data.features
        self.features_norm, self.mean, self.std = normalize_features(
            self.features
        )

        test_features, self.test_labels = generate_mixed_dataset(
            station, n_normal=100, n_attack=30, seed=seed + 1000
        )
        self.test_features_norm = (
            (test_features - self.mean) / self.std
        ).astype(np.float32)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        # Training configuration is pushed by the server's ConfigRecord;
        # module constants are the standalone fallback.
        lr = float(config.get("learning-rate", LEARNING_RATE))
        local_epochs = int(config.get("local-epochs", LOCAL_EPOCHS))

        X = torch.tensor(self.features_norm, dtype=torch.float32)
        dataset = TensorDataset(X, X)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for _ in range(local_epochs):
            for batch_x, _ in loader:
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

        return get_parameters(self.model), len(self.features), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)

        X_test = torch.tensor(self.test_features_norm, dtype=torch.float32)
        errors = reconstruction_error(self.model, X_test).numpy()

        normal_errors = errors[self.test_labels == 0]
        threshold = float(np.percentile(normal_errors, 95))

        predictions = (errors > threshold).astype(int)
        tp = int(((predictions == 1) & (self.test_labels == 1)).sum())
        fp = int(((predictions == 1) & (self.test_labels == 0)).sum())
        fn = int(((predictions == 0) & (self.test_labels == 1)).sum())
        tn = int(((predictions == 0) & (self.test_labels == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / len(self.test_labels)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        loss = float(errors.mean())

        return (
            loss,
            len(self.test_labels),
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "fpr": fpr,
                "threshold": threshold,
                "station": self.station,
            },
        )


def make_client(context: Context) -> StationClient:
    """Build the station client for this node from the run context."""
    partition_id = int(context.node_config.get("partition-id", 0))
    station = STATION_NAMES[partition_id % len(STATION_NAMES)]
    seed = 42 + partition_id * 100
    hidden_dim = int(context.run_config.get("hidden-dim", 32))
    latent_dim = int(context.run_config.get("latent-dim", 8))
    return StationClient(
        station, seed, hidden_dim=hidden_dim, latent_dim=latent_dim
    )


app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    client = make_client(context)
    parameters = msg.content["arrays"].to_numpy_ndarrays()
    config = dict(msg.content["config"]) if "config" in msg.content else {}

    _, num_examples, _ = client.fit(parameters, config)

    # state_dict keys are preserved so the server can save the final
    # aggregated arrays back into a torch state dict.
    content = RecordDict(
        {
            "arrays": ArrayRecord(client.model.state_dict()),
            "metrics": MetricRecord({"num-examples": num_examples}),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    client = make_client(context)
    parameters = msg.content["arrays"].to_numpy_ndarrays()
    config = dict(msg.content["config"]) if "config" in msg.content else {}

    loss, num_examples, metrics = client.evaluate(parameters, config)
    numeric_metrics = {
        k: v for k, v in metrics.items() if isinstance(v, (int, float))
    }

    content = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    "eval_loss": loss,
                    "num-examples": num_examples,
                    **numeric_metrics,
                }
            )
        }
    )
    return Message(content=content, reply_to=msg)
