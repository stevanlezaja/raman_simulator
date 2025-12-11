import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import raman_amplifier as ra
from utils.loading_data_from_file import load_raman_dataset


class ForwardNN(torch.nn.Module):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(6, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 40),
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

    def fit(self, file_path: str, epochs: int = 200, batch_size: int = 32):
        X, Y = self._prepare_dataset(file_path)
        loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)

        final_loss = None

        for epoch in range(epochs):
            total = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                pred = self.net(xb)
                loss = self.loss_fn(pred, yb)
                loss.backward()
                self.optimizer.step()
                total += loss.item()

            final_loss = total / len(loader)
            print(f"[ForwardNN] epoch {epoch+1}/{epochs}, loss={final_loss:.6f}")

        return final_loss

    def forward(self, x):
        return self.net(x)
