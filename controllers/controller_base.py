from abc import ABC, abstractmethod

from custom_types import Power, Length
from raman_amplifier import Pump


class _Controller(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def get_control(self, curr_input: dict[Pump, tuple[Power, Length]], target_output) -> dict[Pump, tuple[Power, Length]]:
        raise NotImplementedError