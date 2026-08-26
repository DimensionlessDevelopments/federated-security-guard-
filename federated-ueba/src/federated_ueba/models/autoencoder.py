import torch.nn as nn


class SecurityAutoencoder(nn.Module):
    """
    Small autoencoder for behavioural anomaly detection.

    Input:
        Normalized security-event feature vector

    Output:
        Reconstruction of the input vector
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        latent_dim: int = 8,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)