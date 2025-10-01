"""
Module for modeling a Raman Amplifier with forward and backward pumps.

This module defines classes to represent a Raman Amplifier, its pumps,
and validates parameters like pumping ratio.
"""

from typing import Optional

import custom_types as ct


class Pump:
    """
    Represents a Raman pump with a power and wavelength.

    Attributes:
        power (Optional[Power]): The power of the pump.
        wavelength (Optional[Length]): The wavelength of the pump.
    """
    def __init__(self, power: Optional[ct.Power] = None, wavelenght: Optional[ct.Length] = None):
        """
        Initialize a Pump object.

        Args:
            power (Optional[Power]): Initial pump power. Defaults to None.
            wavelenght (Optional[Length]): Initial pump wavelength. Defaults to None.
        """
        self._wavelength = wavelenght
        self._power = power

    @property
    def power(self) -> ct.Power:
        """
        Power of the pump.

        Returns:
            Power: The current pump power.
        """
        assert isinstance(self._power, ct.Power)
        return self._power

    @power.setter
    def power(self, new: ct.Power) -> None:
        """
        Set a new pump power.

        Args:
            new (Power): New pump power.
        """
        assert isinstance(new, ct.Power)
        self._power = new

    @property
    def wavelength(self) -> ct.Length:
        """
        Wavelength of the pump.

        Returns:
            Wavelength: The current pump wavelength.
        """
        assert isinstance(self._wavelength, ct.Length)
        return self._wavelength

    @wavelength.setter
    def wavelength(self, new: ct.Length) -> None:
        """
        Set a new pump wavelength.

        Args:
            new (Wavelength): New pump wavelength.
        """
        assert isinstance(new, ct.Length)
        self._wavelength = new

    @property
    def is_valid(self) -> bool:
        return (self.power.value > 0) ^  (self.wavelength.value > 0)


def _validate_ratio(pumping_ratio: float):
    """
    Validate that the pumping ratio is within the allowed range [0.0, 1.0].

    Args:
        pumping_ratio (float): The pumping ratio to validate.

    Raises:
        ValueError: If the pumping ratio is outside [0.0, 1.0].
    """
    if pumping_ratio < 0 or pumping_ratio > 1.0:
        raise ValueError("Pumping ratio can only be in range [0.0, 1.0]")


class RamanAmplifier:
    """
    Represents a Raman Amplifier with forward and backward pumps.

    Attributes:
        _pumping_ratio (float): Fraction of power allocated to the forward pump.
        _pump_power (Power): Base power for the amplifier pumps.
        pump_wavelength (Length): Wavelength of the pump.
        forward_pump (Pump): Forward pump of the amplifier.
        backward_pump (Pump): Backward pump of the amplifier.
    """
    def __init__(self, pumping_ratio: float=1.0):
        """
        Initialize a RamanAmplifier with a given pumping ratio.

        Args:
            pumping_ratio (float, optional): Fraction of power for forward pump.
                Must be in [0.0, 1.0]. Defaults to 1.0.
        """
        _validate_ratio(pumping_ratio)
        self._pumping_ratio = pumping_ratio
        self._pump_power = ct.Power(0.5, 'W')
        self.pump_wavelength = ct.Length(1455, 'nm')
        self.forward_pump = Pump()
        self.backward_pump = Pump()
        self.update_pumps()

    def update_pumps(self) -> None:
        """
        Update the forward and backward pump powers based on the current
        pumping ratio and base pump power.
        """
        self.forward_pump.power = ct.Power(self._pump_power.W * self._pumping_ratio, 'W')
        self.backward_pump.power = ct.Power(self._pump_power.W * (1 - self._pumping_ratio), 'W')
        self.forward_pump.wavelength = self.pump_wavelength
        self.backward_pump.wavelength = self.pump_wavelength

    @property
    def pump_power(self) -> ct.Power:
        """
        Power of the amplifier pumps.

        Returns:
            Power: The current base pump power.
        """
        return self._pump_power

    @pump_power.setter
    def pump_power(self, new: ct.Power) -> None:
        """
        Set a new base pump power and update forward/backward pump powers.

        Args:
            new (Power): New base pump power.
        """
        if new == self._pump_power:
            return
        self._pump_power = new
        self.update_pumps()

    @property
    def pumping_ratio(self) -> float:
        """
        Fraction of total power allocated to the forward pump.

        Returns:
            float: Current pumping ratio in [0.0, 1.0].
        """
        return self._pumping_ratio

    @pumping_ratio.setter
    def pumping_ratio(self, new: float) -> None:
        """
        Set a new pumping ratio and update forward/backward pump powers.

        Args:
            new (float): New pumping ratio. Must be in [0.0, 1.0].

        Raises:
            ValueError: If the new pumping ratio is outside [0.0, 1.0].
        """
        if new == self._pumping_ratio:
            return
        _validate_ratio(new)
        self._pumping_ratio = new
        self.update_pumps()

    @property
    def is_valid(self) -> bool:
        valid = self.backward_pump.is_valid ^ self.forward_pump.is_valid
        return valid
