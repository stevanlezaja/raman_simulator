"""
Module containing the implementation of the RamanSystem class.

This module defines the RamanSystem, which represents a Raman amplification system
consisting of:
    - A Raman Amplifier with an arbitrary number of forward and backward pump pairs
    - A connecting Fiber
    - Input and Output Power Spectra
    - Raman Inputs for the amplifiers

The RamanSystem provides properties to access and set the amplifier, fiber, and spectra,
and an `update` method to propagate the input spectrum through the system.
"""

from typing import Optional

import raman_amplifier as ra
import custom_types as ct
import custom_types.conversions as conv
import fibers as fib
import signals as sig
import experiment as exp
import custom_logging as clog


log = clog.get_logger("Raman System")


class RamanSystem:
    """
    Represents a Raman amplification system.

    Attributes:
        _raman_amplifier (Optional[ra.RamanAmplifier]): The Raman amplifier instance.
        _fiber (Optional[fib.Fiber]): The fiber connecting the pumps.
        _input_spectrum (Optional[ra.Spectrum[ct.Power]]): Input power spectrum.
        _output_spectrum (Optional[ra.Spectrum[ct.Power]]): Output power spectrum.
        _raman_inputs (Optional[ra.RamanInputs]): Raman inputs for the amplifier.

    The class ensures that all components are initialized before use, raising
    AttributeError if an uninitialized property is accessed.
    """

    def __init__(self) -> None:
        """
        Initialize an empty RamanSystem instance.

        All attributes are initialized to None. They should be set via the provided
        property setters before calling the `update` method.
        """

        self._raman_amplifier: Optional[ra.RamanAmplifier] = None
        self._fiber: Optional[fib.Fiber] = None
        self._input_spectrum: Optional[ra.Spectrum[ct.Power]] = None
        self._output_spectrum: Optional[ra.Spectrum[ct.Power]] = None
        self._raman_inputs: Optional[ra.RamanInputs] = None

    @property
    def raman_amplifier(self):
        """
        Get the Raman amplifier instance.

        Returns:
            ra.RamanAmplifier: The initialized Raman amplifier.

        Raises:
            AttributeError: If the Raman amplifier has not been initialized.
        """

        if self._raman_amplifier is None:
            log.error(
                "Attempted to access uninitialized property '%s' on %s",
                "raman_amplifier",
                self.__class__.__name__,
            )
            raise AttributeError(
                f"Property 'raman_amplifier' on {self.__class__.__name__} "
                f"was accessed before initialization"
            )
        return self._raman_amplifier

    @raman_amplifier.setter
    def raman_amplifier(self, new: ra.RamanAmplifier):
        """
        Set the Raman amplifier instance.

        Args:
            new (ra.RamanAmplifier): The Raman amplifier to assign.
        """

        self._raman_amplifier = new

    @property
    def raman_inputs(self):
        """
        Get the system's Raman inputs.

        Returns:
            ra.RamanInputs: current set of input values.

        Raises:
            AttributeError: If there are no Raman Inputs.
        """

        if self._raman_inputs is None:
            log.error(
                "Attempted to access uninitialized property '%s' on %s",
                "raman_inputs",
                self.__class__.__name__,
            )
            raise AttributeError(
                f"Property 'raman_inputs' on {self.__class__.__name__} "
                f"was accessed before initialization"
            )
        return self._raman_inputs

    @raman_inputs.setter
    def raman_inputs(self, new: ra.RamanInputs):
        """
        Set the Raman inputs. and updates the pumps

        Args:
            new (ra.RamanInputs): The Raman inputs to assign.
        """

        self._raman_inputs = new
        self.raman_amplifier.pump_powers = new.powers
        self.raman_amplifier.pump_wavelengths = new.wavelengths

    @property
    def fiber(self):
        """
        Get the fiber connecting the pumps.

        Returns:
            fib.Fiber: The initialized fiber.

        Raises:
            AttributeError: If the fiber has not been initialized.
        """

        if self._fiber is None:
            log.error(
                "Attempted to access uninitialized property '%s' on %s",
                "fiber",
                self.__class__.__name__,
            )
            raise AttributeError(
                f"Property 'fiber' on {self.__class__.__name__} "
                f"was accessed before initialization"
            )
        return self._fiber

    @fiber.setter
    def fiber(self, new: fib.Fiber):
        """
        Set the fiber connecting the pumps.

        Args:
            new (fib.Fiber): The fiber to assign.
        """
        self._fiber = new

    @property
    def input_spectrum(self) -> ra.Spectrum[ct.Power]:
        """
        Get the input power spectrum.

        Returns:
            ra.Spectrum[ct.Power]: The initialized input spectrum.

        Raises:
            AttributeError: If the input spectrum has not been initialized.
        """

        if self._input_spectrum is None:
            log.error(
                "Attempted to access uninitialized property '%s' on %s",
                "input_spectrum",
                self.__class__.__name__,
            )
            raise AttributeError(
                f"Property 'input_spectrum' on {self.__class__.__name__} "
                f"was accessed before initialization"
            )
        return self._input_spectrum

    @input_spectrum.setter
    def input_spectrum(self, new: ra.Spectrum[ct.Power]):
        """
        Set the input power spectrum.

        Args:
            new (ra.Spectrum[ct.Power]): The input spectrum to assign.
        """

        self._input_spectrum = new

    @property
    def output_spectrum(self):
        """
        Get the output power spectrum.

        Returns:
            ra.Spectrum[ct.Power]: The initialized output spectrum.

        Raises:
            AttributeError: If the output spectrum has not been initialized.
        """

        if self._output_spectrum is None:
            log.error(
                "Attempted to access uninitialized property '%s' on %s",
                "output_spectrum",
                self.__class__.__name__,
            )
            raise AttributeError(
                f"Property 'output_spectrum' on {self.__class__.__name__} "
                f"was accessed before initialization"
            )
        return self._output_spectrum

    @output_spectrum.setter
    def output_spectrum(self, new: ra.Spectrum[ct.Power]):
        """
        Set the output power spectrum.

        Args:
            new (ra.Spectrum[ct.Power]): The output spectrum to assign.
        """

        self._output_spectrum = new

    def update(self) -> None:
        """
        Propagate the input spectrum through the Raman system.

        Iterates over each frequency-power pair in the input spectrum, generates a
        signal, runs an experiment using the current fiber and amplifier, and stores
        the resulting power at the fiber output in the output spectrum.

        Raises:
            AttributeError: If any required property (input_spectrum, fiber, or
                            raman_amplifier) is not initialized.
        """

        for input_component in self.input_spectrum:
            freq, power = input_component
            signal = sig.Signal()
            signal.power = power
            signal.wavelength = conv.frequency_to_wavelenth(freq)
            self.output_spectrum[freq] = ct.Power(0, 'W')

            experiment = exp.Experiment(self.fiber, signal, self.raman_amplifier.pump_pairs)

            self.output_spectrum[freq] = experiment.get_signal_power_at_distance(self.fiber.length)

    @property
    def is_valid(self) -> bool:
        valid = self.fiber.is_valid ^ self.raman_amplifier.is_valid
        return valid
