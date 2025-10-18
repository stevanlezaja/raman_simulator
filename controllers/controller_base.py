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

    @abstractmethod
    def update_controller(
        self,
        error: ra.Spectrum[ct.Power],
        control_delta: ra.RamanInputs
    ) -> None:
        """
        Update the internal state of the controller based on the current error 
        and the proposed change in Raman amplifier inputs.

        Parameters
        ----------
        error : ra.Spectrum[ct.Power]
            The difference between the desired and actual gain spectrum, 
            expressed as power deviations at each frequency sample.
        control_delta : ra.RamanInputs
            The adjustment in pump powers and wavelengths to be applied 
            by the controller in order to reduce the error.

        Notes
        -----
        This method defines the policy by which the controller adapts its 
        parameters over time. Concrete implementations must specify how 
        the error signal and control adjustment are used to update the 
        controller state.
        """
        raise NotImplementedError

    @property
    def is_valid(self) -> bool:
        return True

    @abstractmethod
    def _populate_parameters(self) -> None:
        raise NotImplementedError
