import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

import raman_amplifier as ra
from utils.loading_data_from_file import load_raman_dataset


class BackwardNN(torch.nn.Module):
    """
    Physics-consistent inverse Raman model:
        GainSpectrum -> RamanInputs
    """

    def __init__(
        self,
        forward_model: torch.nn.Module,
        lr: float = 1e-3,
        lambda_param: float = 1e-3,
        *args, **kwargs,
    ):
        super().__init__()

        self.forward_model = forward_model
        self.lambda_param = lambda_param

        # ---- Inverse network (LOW capacity on purpose)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(40, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 6),
        )

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )

        self.mse = torch.nn.MSELoss()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.eval()
        print(f"[BackwardNN] loaded model from {path}")

    def _prepare_dataset(self, file_path: str):
        samples = list(load_raman_dataset(file_path))

        # ---- spectrum normalization limits
        all_vals = []
        for _, spec in samples:
            arr = spec.as_array()
            all_vals.append(arr[len(arr) // 2:])

        all_vals = np.vstack(all_vals)

        ra.Spectrum.set_normalization_limits(
            min_val=float(all_vals.min()),
            max_val=float(all_vals.max()),
        )

        X, Y = [], []

        for raman_inputs, spectrum in samples:
            y = torch.tensor(
                raman_inputs.normalize().as_array(),
                dtype=torch.float32,
            )

            arr = spectrum.normalize().as_array()
            x = torch.tensor(arr[len(arr) // 2:], dtype=torch.float32)

            X.append(x)
            Y.append(y)

        return torch.stack(X), torch.stack(Y)

    def _constrain_outputs(self, y_hat):
        """
        Enforce physical bounds:
            pump powers ∈ [0, 1]
            wavelengths ∈ [0, 1] (still normalized)
        """
        return torch.sigmoid(y_hat)

    def fit(
        self,
        training_data_path: str,
        epochs: int = 100,
        batch_size: int = 32,
        val_ratio: float = 0.2,
        plot_losses: bool = False,
        *args, **kwargs
    ):
        X, Y = self._prepare_dataset(training_data_path)
        dataset = TensorDataset(X, Y)

        n_val = int(len(dataset) * val_ratio)
        n_train = len(dataset) - n_val

        train_ds, val_ds = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        train_losses, val_losses = [], []

        best_val = float("inf")

        for _ in tqdm(range(epochs)):
            self.train()
            train_loss = 0.0

            for x, y_true in train_loader:
                self.optimizer.zero_grad()

                y_hat = self._constrain_outputs(self.net(x))

                spec_hat = self.forward_model(y_hat)

                loss_spec = self.mse(spec_hat, x)
                loss_param = self.mse(y_hat, y_true)

                loss = loss_spec + self.lambda_param * loss_param

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            self.eval()
            val_loss = 0.0

            with torch.no_grad():
                for x, y_true in val_loader:
                    y_hat = self._constrain_outputs(self.net(x))
                    spec_hat = self.forward_model(y_hat)

                    loss = self.mse(spec_hat, x)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss

        if plot_losses:
            self._plot_losses(train_losses, val_losses)

        return best_val

    def forward(self, x):
        return self._constrain_outputs(self.net(x))

    def _plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Training loss")
        plt.plot(val_losses, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE (spectrum)")
        plt.title("BackwardNN (Physics-consistent)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
