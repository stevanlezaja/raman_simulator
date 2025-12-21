import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.utils.data import random_split
import matplotlib.pyplot as plt

import raman_amplifier as ra
from utils.loading_data_from_file import load_raman_dataset


class ForwardNN(torch.nn.Module):
    def __init__(self, lr: float = 1e-3, *args, **kwargs):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(6, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 40),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.eval()
        print(f"[ForwardNN] loaded model from {path}")

    def _prepare_dataset(self, file_path: str):
        """Load dataset, compute normalization limits, produce torch tensors"""

        samples = list(load_raman_dataset(file_path))

        all_spec_vals = []
        for _, spec in samples:
            arr = spec.as_array()
            all_spec_vals.append(arr[len(arr) // 2:])

        all_spec_vals = np.vstack(all_spec_vals)
        ra.Spectrum.set_normalization_limits(
            min_val=float(all_spec_vals.min()),
            max_val=float(all_spec_vals.max())
        )

        X_list = []
        Y_list = []

        for raman_inputs, spectrum in samples:
            x = torch.tensor(raman_inputs.normalize().as_array(), dtype=torch.float32)

            arr = spectrum.normalize().as_array()
            values = arr[len(arr) // 2:]
            y = torch.tensor(values, dtype=torch.float32)

            X_list.append(x)
            Y_list.append(y)

        X = torch.stack(X_list)
        Y = torch.stack(Y_list)
        return X, Y

    def _plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Training loss")
        plt.plot(val_losses, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.title("ForwardNN Training")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def fit(
        self,
        file_path: str,
        epochs: int = 200,
        batch_size: int = 32,
        val_ratio: float = 0.2,
        plot_losses: bool = True,
    ):
        X, Y = self._prepare_dataset(file_path)
        dataset = TensorDataset(X, Y)

        n_total = len(dataset)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val

        train_ds, val_ds = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        train_losses = []
        val_losses = []

        best_val_loss = float("inf")

        for _ in tqdm(range(epochs)):
            # ===== TRAIN =====
            self.train()
            train_loss = 0.0

            for xb, yb in train_loader:
                self.optimizer.zero_grad()
                pred = self.net(xb)
                loss = self.loss_fn(pred, yb)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # ===== VALIDATE =====
            self.eval()
            val_loss = 0.0

            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = self.net(xb)
                    loss = self.loss_fn(pred, yb)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # self.save("forward_nn_best.pt")

        if plot_losses:
            self._plot_losses(train_losses, val_losses)

        return best_val_loss


    def forward(self, x):
        return self.net(x)
