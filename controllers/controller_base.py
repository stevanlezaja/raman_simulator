from abc import ABC, abstractmethod

import raman_amplifier as ra
import custom_types as ct


class Controller(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def get_control(self, curr_input: ra.RamanInputs, curr_output:ra.Spectrum[ct.Power], target_output: ra.Spectrum[ct.Power]) -> ra.RamanInputs:
        raise NotImplementedError
