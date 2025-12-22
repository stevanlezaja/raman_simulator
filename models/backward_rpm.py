import random
from typing import List
import torch
import torch.nn as nn


class BackwardRPM(nn.Module):
    def __init__(
        self,
        forward_model: nn.Module,
        proj_dim: int = 8,
        steps: int = 200,
        lr: float = 1e-2,
        clamp: bool = True,
    ):
        super().__init__()

        self.forward_model = forward_model
        self.steps = steps
        self.lr = lr
        self.clamp = clamp

        # Random projection matrix (fixed)
        R = torch.randn(proj_dim, 40)
        R /= torch.norm(R, dim=1, keepdim=True)

        self.register_buffer("R", R)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        spectrum: normalized gain spectrum (40,)
        returns: normalized Raman inputs (6,)
        """

        # Initialize u
        u = torch.full((6,), 0.5, requires_grad=False)

        for _ in range(self.steps):
            g_pred = self.forward_model(u)
            error = g_pred - spectrum

            z = self.R @ error               # (proj_dim,)
            grad_est = self.R.T @ z          # (40,)
            grad_u = grad_est[:6]            # map to input space

            u = u - self.lr * grad_u

            if self.clamp:
                u = torch.clamp(u, 0.0, 1.0)

        return u.detach()


class BackwardEnsemble(torch.nn.Module):
    def __init__(self, model_paths: List[str]):
        """
        Wrap multiple RPM models as an ensemble.
        :param model_paths: list of file paths to saved RPM model weights
        """
        super().__init__()
        self.models = []
        for path in model_paths:
            model = torch.load(path)  # or your load function
            model.eval()  # ensure evaluation mode
            self.models.append(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Randomly pick one model for each forward call
        model = random.choice(self.models)
        return model(x)
