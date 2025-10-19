"""
    Base module for Controller objects
"""

from abc import ABC, abstractmethod
from typing import Any

import utils.parameter
import raman_amplifier as ra
import custom_types as ct


class Controller(ABC):
    """
    Abstract base class for controllers of Raman Systems
        Defines the get_control method
    """
    def __init__(self):
        self._params: dict[str, tuple[type, Any]] = {}

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

    def populate_parameters(self, value_dict: dict[str, Any] = {}) -> None:
        """
        Populate controller parameters from a provided dictionary or via user input.

        This method iterates through all controller parameters defined in
        ``self._params``. Each parameter entry is a tuple of the form
        ``(type, value)`` where ``type`` denotes the expected parameter type
        (e.g., ``float`` or a subclass of ``ct.units.Unit``) and ``value`` is
        the current or default value.

        If a parameter key exists in ``value_dict``, its value is assigned
        directly. Otherwise, the user is interactively prompted to input
        a new value through the appropriate utility function:
        
        - ``utils.parameter.get_numeric_input()`` for numeric parameters
        - ``utils.parameter.get_unit_input()`` for unit-based parameters

        Parameters
        ----------
        value_dict : dict[str, Any], optional
            A dictionary mapping parameter names to their desired values.
            Missing keys trigger interactive input prompts. Defaults to an empty dict.

        Raises
        ------
        Exception
            If a parameter type is not ``float`` or a subclass of
            ``ct.units.Unit``, an exception is raised.

        Notes
        -----
        This method modifies ``self._params`` in place, ensuring that all
        required controller parameters are populated either from external
        configuration data or user input.
        """

        for key in self._params.keys():
            param_type, param = self._params[key]
            print(param_type)
            if key in value_dict.keys():
                self._params[key] = value_dict[key]
            elif param_type == float:
                self._params[key] = (param_type, utils.parameter.get_numeric_input(f"Please input {key}", param))
            elif issubclass(param_type, ct.units.Unit):
                self._params[key] = (param_type, utils.parameter.get_unit_input(param, param, key))
            else:
                raise Exception(f"Unhandled parameter type: {param_type}")
