import torch
from torch import nn


class RandomProjectionInverseModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 40,
        output_dim: int = 6,
        hidden_dim: int = 800,
        n_layers: int = 7,
        sigma_rpm: float = 0.1,
        random_seed: int | None = None,
        activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()

        if random_seed is not None:
            torch.manual_seed(random_seed)

        self.activation = activation
        self.sigma_rpm = sigma_rpm
        self.random_layers = nn.ModuleList()

        self.random_layers.append(nn.Linear(input_dim, hidden_dim, bias=True))
        for _ in range(n_layers - 1):
            self.random_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))

        # --- Zibar-style Gaussian initialization ---
        for layer in self.random_layers:
            nn.init.normal_(layer.weight, mean=0.0, std=self.sigma_rpm)
            nn.init.zeros_(layer.bias)
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False

        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.random_layers:
            h = self.activation(layer(h))
        return h

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        h = self.forward_hidden(x)
        return self.output_layer(h)

    @torch.no_grad()
    def fit_closed_form(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        lambda_reg: float = 1e-4,
    ):
        """
        One-shot Random Projection Method training
        """
        self.eval()

        H = self.forward_hidden(X_train)
        d = H.shape[1]

        I = torch.eye(d, device=H.device)
        A = H.T @ H + lambda_reg * I
        B = H.T @ Y_train

        W = torch.linalg.solve(A, B)

        self.output_layer.weight.copy_(W.T)
        self.output_layer.bias.zero_()
