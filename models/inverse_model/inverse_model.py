from copy import deepcopy
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import custom_types as ct
import raman_amplifier as ra
from utils.loading_data_from_file import load_raman_dataset

from .random_projection_model import RandomProjectionInverseModel

class InverseModel:
    def __init__(self) -> None:
        self.models = [RandomProjectionInverseModel(), RandomProjectionInverseModel(), RandomProjectionInverseModel(), 
                        RandomProjectionInverseModel(), RandomProjectionInverseModel(), RandomProjectionInverseModel(), 
                        RandomProjectionInverseModel(), RandomProjectionInverseModel(), RandomProjectionInverseModel(), 
                        RandomProjectionInverseModel()]
        X, Y = self._prepare_training_tensors('data/raman_simulator/3_pumps/100_fiber_0.0_ratio_sorted.json')
        batch_size = 8
        self.loss_history = []

        for model in self.models:
            epoch_losses = []
            for epoch in tqdm(range(4)):
                batch_losses = []

                for i in range(0, len(X), batch_size):
                    X_batch = X[i:i + batch_size]
                    Y_batch = Y[i:i + batch_size]

                    loss = model.fit(
                        epochs=1,
                        X_train=X_batch,
                        Y_train=Y_batch
                    )

                    batch_losses.append(loss)

                epoch_loss = sum(batch_losses) / len(batch_losses)
                epoch_losses.append(epoch_loss)
            self.loss_history = epoch_losses
            # self.plot_loss()

    def plot_loss(self):
        plt.figure()
        plt.plot(self.loss_history)
        plt.xlabel("Training step (batch)")
        plt.ylabel("Loss")
        plt.title("Inverse Model Training Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_raman_inputs(self, spectrum: ra.Spectrum[ct.Power]) -> ra.RamanInputs:
        assert isinstance(spectrum, ra.Spectrum)
        spectrum_copy = deepcopy(spectrum)
        spectrum_arr = spectrum_copy.normalize().as_array()

        raman_inputs_arr_list = [model.forward(spectrum_arr[len(spectrum_arr)//2:]).detach().numpy() for model in self.models]
        raman_inputs_arr = np.mean(np.stack(raman_inputs_arr_list, axis=0), axis=0)
        raman_inputs = ra.RamanInputs.from_array(raman_inputs_arr).denormalize()
        return raman_inputs
    
    def _prepare_training_tensors(self, dataset_path: str):
        def _compute_spectrum_norm(dataset_path):
            min_val = float('inf')
            max_val = float('-inf')
            for _, spectrum in load_raman_dataset(dataset_path):
                arr = deepcopy(spectrum).as_array()
                min_val = min(min_val, arr.min())
                max_val = max(max_val, arr.max())
            return min_val, max_val
        norm_min, norm_max = _compute_spectrum_norm('data/raman_simulator/3_pumps/100_fiber_0.0_ratio_sorted.json')
        ra.Spectrum.norm_min = norm_min
        ra.Spectrum.norm_max = norm_max

        X_list = []
        Y_list = []

        for raman_inputs, spectrum in load_raman_dataset(dataset_path):
            x_arr = spectrum.normalize().as_array()
            X_list.append(x_arr[len(x_arr)//2:])

            y_arr = raman_inputs.normalize().as_array()
            Y_list.append(y_arr)

        X_train = torch.tensor(X_list, dtype=torch.float32)
        Y_train = torch.tensor(Y_list, dtype=torch.float32)

        return X_train, Y_train
