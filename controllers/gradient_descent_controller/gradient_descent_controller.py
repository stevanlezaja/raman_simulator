import torch

import raman_amplifier as ra
import custom_types as ct
from utils.loading_data_from_file import load_raman_dataset

from ..controller_base import Controller
from .forward_nn import ForwardNN

class GradientDescentController(Controller, torch.nn.Module):
    def __init__(self):
        pass

    def train_controller(self, file_path: str):
        pass

    def get_control(self, curr_input: RamanInputs, curr_output: Spectrum[Power], target_output: Spectrum[Power]) -> RamanInputs:
        pass

    def update_controller(self, error: Spectrum[Power], control_delta: RamanInputs) -> None:
        pass