"""
    Base module for Controller objects
"""

from abc import ABC, abstractmethod

import raman_amplifier as ra
import custom_types as ct


class Controller(ABC):
    """
    Abstract base class for controllers of Raman Systems
        Defines the get_control method
    """
    def __init__(self):
        pass

    @abstractmethod
    def get_control(
        self,
        curr_input: ra.RamanInputs,
        curr_output:ra.Spectrum[ct.Power],
        target_output: ra.Spectrum[ct.Power],
    ) -> ra.RamanInputs:
        """
        get_control abstract method to be implemented by all controller classes
            inputs:
                curr_input - last created RamanInput object
                curr_output - last measured Spectrum object
                target_output - target Spectrum object
            outputs:
                RamanInputs - new control signal to be applied
        """
        raise NotImplementedError
