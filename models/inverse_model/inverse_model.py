from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore

import custom_types as ct
import raman_amplifier as ra
from utils.loading_data_from_file import load_raman_dataset
from .random_projection_model import RandomProjectionInverseModel


class InverseModel:
    def __init__(self, n_models=200, hidden_dim=800, n_layers=7, sigma_rpm=0.2) -> None:  # type: ignore
        self.models = [
            RandomProjectionInverseModel(
                input_dim=40,
                output_dim=6,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                sigma_rpm=sigma_rpm,
                activation=nn.Tanh()
            )
            for _ in range(n_models)
        ]
        self.train_loss_history = [[] for _ in range(n_models)]  # type: ignore
        self.val_loss_history = [[] for _ in range(n_models)]  # type: ignore
        self.sigma_rpm = sigma_rpm

        X, Y = self._prepare_training_tensors('data/raman_simulator/3_pumps/100_fiber_0.0_ratio_sorted.json')
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)  # type: ignore

        self.X_train: torch.Tensor = X_train
        self.Y_train: torch.Tensor = Y_train
        self.X_val: torch.Tensor = X_val
        self.Y_val: torch.Tensor = Y_val

        for idx, model in tqdm(enumerate(self.models), total=len(self.models)):
            loaded = self._load_model(model, idx)
            if loaded:
                val_loss = self._validate_model(model)
                self.val_loss_history[idx].append(val_loss)  # type: ignore
                continue
            model.fit_closed_form(
                self.X_train,  # type: ignore
                self.Y_train,  # type: ignore
                lambda_reg=1e-4,
            )
            val_loss = self._validate_model(model)
            self.train_loss_history[idx].append(float("nan"))  # type: ignore
            self.val_loss_history[idx].append(val_loss)  # type: ignore

            self._save_model(model, idx)

    @torch.no_grad()  # type: ignore
    def _validate_model(
        self,
        model: nn.Module,
    ) -> float:
        model.eval()
        pred = model(self.X_val)
        return float(torch.mean((pred - self.Y_val) ** 2).item())

    def _save_model(self, model: nn.Module, model_idx: int):
        path = self._model_path(model_idx)
        torch.save(model.state_dict(), path)

    def _load_model(self, model: nn.Module, model_idx: int) -> bool:
        path = self._model_path(model_idx)
        if path.exists():
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            return True
        return False

    def _model_path(self, model_idx: int) -> Path:
        base_dir = Path("models/models/rpm_inverse") / f"sigma_{self.sigma_rpm:.3f}"
        base_dir.mkdir(parents=True, exist_ok=True)

        activation_name = type(self.models[model_idx].activation).__name__

        filename = (
            f"rpm_inv_"
            f"sigma{self.sigma_rpm:.3f}_"
            f"ensemble{len(self.models)}_"
            f"idx{model_idx}_"
            f"in40_out6_"
            f"h{self.models[model_idx].output_layer.in_features}_"
            f"layers{len(self.models[model_idx].random_layers)}_"
            f"act{activation_name}_"
            f"dataset3pumps.pt"
        )
        return base_dir / filename

    def get_raman_inputs(self, spectrum: ra.Spectrum[ct.Power]) -> ra.RamanInputs:
        assert isinstance(spectrum, ra.Spectrum)
        spectrum_copy = deepcopy(spectrum)
        spectrum_arr = spectrum_copy.normalize().as_array(include_freq=False)

        raman_inputs_arr_list = [
            model.forward(spectrum_arr).detach().numpy() for model in self.models  # type: ignore
        ]
        mean_inputs = np.mean(raman_inputs_arr_list, axis=0)
        return ra.RamanInputs.from_array(mean_inputs).denormalize()

    def _prepare_training_tensors(self, dataset_path: str):
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
