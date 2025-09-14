from abc import ABC, abstractmethod

from raman_amplifier import RamanInputs, GainSpectrum


class _Controller(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def get_control(self, curr_input: RamanInputs, curr_output:GainSpectrum, target_output: GainSpectrum) -> RamanInputs:
        raise NotImplementedError
