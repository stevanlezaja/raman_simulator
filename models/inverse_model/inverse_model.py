from copy import deepcopy
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import custom_types as ct
import raman_amplifier as ra
from utils.loading_data_from_file import load_raman_dataset
from .random_projection_model import RandomProjectionInverseModel

class InverseModel:
    def __init__(self, n_models=10, hidden_dim=800, n_layers=7) -> None:
        # Create ensemble of RPM models
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
        self.loss_history = [[] for _ in range(n_models)]

        # Prepare training tensors
        X, Y = self._prepare_training_tensors(
            'data/raman_simulator/3_pumps/100_fiber_0.0_ratio_sorted.json'
        )

        # Train each model
        batch_size = 64
        for idx, model in enumerate(tqdm(self.models, desc="Training RPM ensemble")):
            self._train_model(model, X, Y, batch_size=batch_size, model_idx=idx, epochs=10)

    def _train_model(self, model, X_train, Y_train, batch_size=64, model_idx=0, epochs=20, lr=1e-3):
        optimizer = torch.optim.Adam(model.output_layer.parameters(), lr=lr)
        criterion = nn.MSELoss()
        n_samples = len(X_train)

        for epoch in range(epochs):
            perm = torch.randperm(n_samples)
            X_train_shuffled = X_train[perm]
            Y_train_shuffled = Y_train[perm]

            batch_losses = []
            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                Y_batch = Y_train_shuffled[i:i+batch_size]

                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, Y_batch)
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            epoch_loss = float(np.mean(batch_losses))
            self.loss_history[model_idx].append(epoch_loss)

        # # Plot epoch losses, not batch losses
        # plt.figure()
        # plt.plot(self.loss_history[model_idx])
        # plt.xlabel("Epoch")
        # plt.ylabel("MSE Loss")
        # plt.title(f"Model {model_idx+1} Training Loss")
        # plt.show()


    def plot_loss(self):
        plt.figure(figsize=(8, 5))
        for idx, model_losses in enumerate(self.loss_history):
            plt.plot(model_losses, label=f'Model {idx+1}')
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("RPM Ensemble Training Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

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
        def _compute_spectrum_norm(dataset_path):
            min_val = float('inf')
            max_val = float('-inf')
            for _, spectrum in load_raman_dataset(dataset_path):
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
            X_list.append(x_arr)

            y_arr = raman_inputs.normalize().as_array()
            Y_list.append(y_arr)

        X_train = torch.tensor(X_list, dtype=torch.float32)
        Y_train = torch.tensor(Y_list, dtype=torch.float32)

        return X_train, Y_train
