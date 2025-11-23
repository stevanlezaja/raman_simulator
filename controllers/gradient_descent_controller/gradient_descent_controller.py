import torch

from custom_types.power import Power
from raman_amplifier.raman_inputs import RamanInputs
from raman_amplifier.spectrum import Spectrum
from ..controller_base import Controller

class GradientDescentController(Controller, torch.nn.Module):
    def __init__(self):
        pass

    def train_controller(self, file_path: str):
        pass

    def get_control(self, curr_input: RamanInputs, curr_output: Spectrum[Power], target_output: Spectrum[Power]) -> RamanInputs:
        pass

    def update_controller(self, error: Spectrum[Power], control_delta: RamanInputs) -> None:
        pass