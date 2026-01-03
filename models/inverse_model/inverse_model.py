from copy import deepcopy
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore

import custom_types as ct
import raman_amplifier as ra
from utils.loading_data_from_file import load_raman_dataset
from .random_projection_model import RandomProjectionInverseModel

class InverseModel:
    def __init__(self, n_models=10, hidden_dim=800, n_layers=7) -> None:  # type: ignore
        self.models = [
            RandomProjectionInverseModel(
                input_dim=40,
                output_dim=6,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                activation=nn.Tanh()
            )
            for _ in range(n_models)
        ]
        self.train_loss_history = [[] for _ in range(n_models)]  # type: ignore
        self.val_loss_history = [[] for _ in range(n_models)]  # type: ignore


        # Prepare training tensors
        X, Y = self._prepare_training_tensors('data/raman_simulator/3_pumps/100_fiber_0.0_ratio_sorted.json')
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)  # type: ignore

        self.X_train: torch.Tensor = X_train
        self.Y_train: torch.Tensor = Y_train
        self.X_val: torch.Tensor = X_val
        self.Y_val: torch.Tensor = Y_val

        # Train each model
        batch_size = 64
        for idx, model in enumerate(tqdm(self.models, desc="Training RPM ensemble")):
            self._train_model(model, batch_size=batch_size, model_idx=idx, epochs=100)

        self.plot_loss()
        self.plot_ensemble_mean_loss()

    def _train_model(
        self,
        model: torch.nn.Module,
        batch_size: int = 64,
        model_idx: int = 0,
        epochs: int = 20,
        lr: float = 1e-3,
    ):
        optimizer = torch.optim.Adam(model.output_layer.parameters(), lr=lr)  # type: ignore
        criterion = nn.MSELoss()

        n_samples = len(self.X_train)

        for _ in range(epochs):
            model.train()
            perm = torch.randperm(n_samples)

            batch_losses = []
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i+batch_size]
                X_batch = self.X_train[idx]
                Y_batch = self.Y_train[idx]

                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, Y_batch)
                loss.backward()
                optimizer.step()  # type: ignore

                batch_losses.append(loss.item())  # type: ignore

            train_loss = float(np.mean(batch_losses))  # type: ignore

            model.eval()
            with torch.no_grad():
                val_pred = model(self.X_val)
                val_loss = criterion(val_pred, self.Y_val).item()

            self.train_loss_history[model_idx].append(train_loss)  # type: ignore
            self.val_loss_history[model_idx].append(val_loss)  # type: ignore


    def plot_loss(self):
        plt.figure(figsize=(8, 5))  # type: ignore
        for i in range(len(self.models)):
            plt.plot(self.train_loss_history[i], alpha=0.6, label=f"Train M{i+1}")  # type: ignore
            plt.plot(self.val_loss_history[i], linestyle="--", alpha=0.6, label=f"Val M{i+1}")  # type: ignore

        plt.xlabel("Epoch")  # type: ignore
        plt.ylabel("MSE Loss")  # type: ignore
        plt.title("RPM Ensemble - Training & Validation Loss")  # type: ignore
        plt.legend(ncol=2, fontsize=8)  # type: ignore
        plt.grid(True)  # type: ignore
        plt.tight_layout()
        plt.show()  # type: ignore


    def plot_ensemble_mean_loss(self):
        train_mean = np.mean(self.train_loss_history, axis=0)  # type: ignore
        train_std = np.std(self.train_loss_history, axis=0)  # type: ignore

        val_mean = np.mean(self.val_loss_history, axis=0)  # type: ignore
        val_std = np.std(self.val_loss_history, axis=0)  # type: ignore

        epochs = np.arange(len(train_mean))

        plt.figure(figsize=(8, 5))  # type: ignore
        plt.plot(epochs, train_mean, label="Train (mean)")  # type: ignore
        plt.fill_between(  # type: ignore
            epochs,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.3,
        )

        plt.plot(epochs, val_mean, label="Validation (mean)")  # type: ignore
        plt.fill_between(  # type: ignore
            epochs,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.3,
        )

        plt.xlabel("Epoch")  # type: ignore
        plt.ylabel("MSE Loss")  # type: ignore
        plt.title("RPM Ensemble - Mean + Std Loss")  # type: ignore
        plt.grid(True)  # type: ignore
        plt.legend()  # type: ignore
        plt.tight_layout()
        plt.show()  # type: ignore


    def get_raman_inputs(self, spectrum: ra.Spectrum[ct.Power]) -> ra.RamanInputs:
        assert isinstance(spectrum, ra.Spectrum)
        spectrum_copy = deepcopy(spectrum)
        spectrum_arr = spectrum_copy.normalize().as_array(include_freq=False)

        # Ensemble predictions
        raman_inputs_arr_list = [
            model.forward(spectrum_arr).detach().numpy() for model in self.models
        ]
        raman_inputs_arr = np.mean(np.stack(raman_inputs_arr_list, axis=0), axis=0)

        return ra.RamanInputs.from_array(raman_inputs_arr).denormalize()

    def _prepare_training_tensors(self, dataset_path: str):
        # Compute normalization constants
        def _compute_spectrum_norm(dataset_path):  # type: ignore
            min_val = float('inf')
            max_val = float('-inf')
            for _, spectrum in load_raman_dataset(dataset_path):  # type: ignore
                arr = deepcopy(spectrum).as_array(include_freq=False)
                min_val = min(min_val, arr.min())
                max_val = max(max_val, arr.max())
            return min_val, max_val

        norm_min, norm_max = _compute_spectrum_norm(dataset_path)
        ra.Spectrum.norm_min = norm_min
        ra.Spectrum.norm_max = norm_max

        X_list = []
        Y_list = []

        for raman_inputs, spectrum in load_raman_dataset(dataset_path):
            # Use full spectrum for input
            x_arr = spectrum.normalize().as_array(include_freq=False)
            X_list.append(x_arr)  # type: ignore

            y_arr = raman_inputs.normalize().as_array()
            Y_list.append(y_arr)  # type: ignore

        X_train = torch.tensor(X_list, dtype=torch.float32)
        Y_train = torch.tensor(Y_list, dtype=torch.float32)

        return X_train, Y_train
