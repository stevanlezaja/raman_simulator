from copy import deepcopy
import torch
from tqdm import tqdm

import custom_types as ct
import raman_amplifier as ra
from utils.loading_data_from_file import load_raman_dataset

from .random_projection_model import RandomProjectionInverseModel

class InverseModel:
    def __init__(self) -> None:
        self.model: RandomProjectionInverseModel = RandomProjectionInverseModel()
        X, Y = self._prepare_training_tensors('data/raman_simulator/3_pumps/100_fiber_0.0_ratio_sorted.json')
        print(X.shape)
        print(Y.shape)
        batch_size = 128
        for i in tqdm(range(0, len(X), batch_size)):
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]
            self.model.fit(epochs=1, X_train=X_batch, Y_train=Y_batch)

    def get_raman_inputs(self, spectrum: ra.Spectrum[ct.Power]) -> ra.RamanInputs:
        assert isinstance(spectrum, ra.Spectrum)
        spectrum_copy = deepcopy(spectrum)
        spectrum_arr = spectrum_copy.normalize().as_array()

        raman_inputs_arr = self.model.forward(spectrum_arr[len(spectrum_arr)//2:]).detach().numpy()
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



# Example