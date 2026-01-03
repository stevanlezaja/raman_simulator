import numpy as np
import torch
import torch.nn as nn
from typing import Union
import matplotlib.pyplot as plt


class RandomProjectionInverseModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 40,
        output_dim: int = 6,
        hidden_dim: int = 800,
        n_layers: int = 7,
        random_seed: int | None = None,
        activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()

        if random_seed is not None:
            torch.manual_seed(random_seed)

        self.activation = activation

        # Fixed random projection layers (Linear but frozen)
        self.random_layers = nn.ModuleList()

        # Input → first hidden
        self.random_layers.append(
            nn.Linear(input_dim, hidden_dim, bias=True)
        )

        # Hidden → hidden
        for _ in range(n_layers - 1):
            self.random_layers.append(
                nn.Linear(hidden_dim, hidden_dim, bias=True)
            )

        # Freeze all random layers
        for layer in self.random_layers:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False

        # Trainable output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        h = x
        for layer in self.random_layers:
            h = self.activation(layer(h))

        return self.output_layer(h)

    def fit(
        self,
        epochs: int,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        lr: float = 1e-3,
        plot: bool = True,
    ) -> float:
        # Only optimize the last layer
        optimizer = torch.optim.Adam(self.output_layer.parameters(), lr=lr)
        criterion = nn.MSELoss()
        losses = []

        for _ in range(epochs):
            optimizer.zero_grad()
            pred = self.forward(X_train)
            loss = criterion(pred, Y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return float(sum(losses) / len(losses))
