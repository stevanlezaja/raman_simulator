import numpy as np
import torch
import torch.nn as nn

class RandomProjectionInverseModel(nn.Module):
    def __init__(self, input_dim=40, output_dim=6, hidden_dim=800, random_seed=None):
        """
        input_dim  : dimension of the gain spectrum (e.g., 40)
        output_dim : number of outputs (pump powers + wavelengths, e.g., 6)
        hidden_dim : number of random hidden neurons
        """
        super().__init__()
        if random_seed is not None:
            torch.manual_seed(random_seed)

        # Fixed random projection weights
        self.random_weights = torch.randn(input_dim, hidden_dim)
        self.random_bias = torch.randn(hidden_dim)

        # Linear output layer (trainable)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Activation function
        self.activation = nn.Tanh()  # or ReLU

    def fit(self, epochs: int, X_train: torch.Tensor, Y_train: torch.Tensor):
        optimizer = torch.optim.Adam(self.output_layer.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        losses = []

        for _ in range(epochs):
            optimizer.zero_grad()
            pred = self.forward(X_train)
            loss = criterion(pred, Y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return sum(losses) / len(losses)


    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        # Random feature projection
        h = self.activation(x @ self.random_weights + self.random_bias)
        # Trainable linear output
        return self.output_layer(h)
