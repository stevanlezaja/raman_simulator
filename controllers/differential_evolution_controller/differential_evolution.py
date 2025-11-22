from custom_types.power import Power
from raman_amplifier import RamanInputs, Spectrum
from ..controller_base import Controller

class DifferentialEvolutionController(Controller):
    def __init__(self):
        super().__init__()

    def get_control(self, curr_input: RamanInputs, curr_output: Spectrum[Power], target_output: Spectrum[Power]) -> RamanInputs:
        return RamanInputs()

    def update_controller(self, error: Spectrum[Power], control_delta: RamanInputs) -> None:
        return