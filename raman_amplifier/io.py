"""
This module contains Input and Output types for a Raman Amplifier
"""

from typing import TypeVar, Generic, Type, Iterator, Optional, Any

import custom_types as ct
import custom_logging as clog


log = clog.get_logger("IO")


class RamanInputs:
    """
    RamanInputs is a class used to represent inputs to a Raman Amplifier
        It contains the Wavelength - Power pairs representing the Pump state
    """

    MAX_POWER_W = 0.75
    MIN_POWER_W = 0.25

    MAX_WAVELENGTH_NM = 1490
    MIN_WAVELENGTH_NM = 1420

    power_range = (ct.Power(MIN_POWER_W, 'W'), ct.Power(MAX_POWER_W, 'W'))
    wavelength_range = (ct.Length(MIN_WAVELENGTH_NM, 'nm'), ct.Length(MAX_WAVELENGTH_NM, 'nm'))

    def __init__(
            self,
            powers: Optional[list[ct.Power]] = None,
            wavelengths: Optional[list[ct.Length]] = None,
            n_pumps: Optional[int] = None
        ):
        if powers is not None or wavelengths is not None:
            assert (
                powers is not None and
                wavelengths is not None
            ), "Both powers and wavelengths need to be provided"

        if powers is None:
            assert n_pumps is not None, "n_pumps cannot be None if powers are None"
            powers = [ct.Power(0.0, 'W') for _ in range(n_pumps)]

        if wavelengths is None:
            assert n_pumps is not None, "n_pumps cannot be None if wavelengths are None"
            wavelengths = [ct.Length(0.0, 'm') for _ in range(n_pumps)]

        self.wavelengths: list[ct.Length] = wavelengths
        self.powers: list[ct.Power] = powers
        self.value_dict: dict[ct.Length, ct.Power] = dict(zip(wavelengths, powers))

    def __add__(self, other: "RamanInputs") -> "RamanInputs":
        new_powers = [p1 + p2 for p1, p2 in zip(self.powers, other.powers)]
        new_wavelengths = [w1 + w2 for w1, w2 in zip(self.wavelengths, other.wavelengths)]
        return RamanInputs(powers=new_powers, wavelengths=new_wavelengths)

    def __sub__(self, other: "RamanInputs") -> "RamanInputs":
        new_powers = [p1 - p2 for p1, p2 in zip(self.powers, other.powers)]
        new_wavelengths = [w1 - w2 for w1, w2 in zip(self.wavelengths, other.wavelengths)]
        return RamanInputs(powers=new_powers, wavelengths=new_wavelengths)

    def clamp_values(self) -> None:
        """Clamp powers and wavelengths to their defined ranges in-place."""

        # Clamp powers
        p_min, p_max = self.power_range
        for i, p in enumerate(self.powers):
            clamped_val = min(max(p.value, p_min.value), p_max.value)
            self.powers[i] = ct.Power(clamped_val, p.default_unit)

        # Clamp wavelengths
        wl_min, wl_max = self.wavelength_range
        for i, wl in enumerate(self.wavelengths):
            clamped_val = min(max(wl.value, wl_min.value), wl_max.value)
            self.wavelengths[i] = ct.Length(clamped_val, wl.default_unit)

        # Update value_dict to stay consistent
        self.value_dict = dict(zip(self.wavelengths, self.powers))

    def __repr__(self):
        return f"Raman inputs: \n Powers: {self.powers},\n Wavelengths: {self.wavelengths}.\n"


T = TypeVar("T", bound = ct.UnitProtocol)

class Spectrum(Generic[T]):
    """
    Spectrum is a generic class that represents values over Frequencies
        Id consists of a spectrum dict that stores the pairs of Frequency and value
    """
    operations = ['+', '-']

    def __init__(self, value_cls: Type[T]):
        self.value_cls: Type[T] = value_cls  # store class type
        self.spectrum: dict[ct.Frequency, T] = {}

    def _linear_op(self, other: "Spectrum[T]", operation: str) -> "Spectrum[T]":
        """
        Method that implements linear operations on Spectrum objects
            Operation is passed as a string and matched to a class variable list
            containing the supported operations.
            When implementing support for a new operation, add its symbol to the list.
        """

        assert operation in Spectrum.operations, f"Operation {operation} is not supported"

        assert self.frequencies and other.frequencies, "There must be frequencies in both spectrums"

        assert (
            isinstance(other, Spectrum)
        ), (f"Both operands need to be type {self.__class__.__name__}")

        assert (
            self.frequencies == other.frequencies
        ), ("Both spectra need to have the same frequencies")

        result: Spectrum[T] = Spectrum(self.value_cls)
        for f, g in self.spectrum.items():
            if operation == '+':
                result.spectrum[f] = g + other.spectrum[f]
            elif operation == '-':
                result.spectrum[f] = g - other.spectrum[f]
        return result

    def __add__(self, other: "Spectrum[T]") -> "Spectrum[T]":
        return self._linear_op(other, '+')

    def __sub__(self, other: "Spectrum[T]") -> "Spectrum[T]":
        return self._linear_op(other, '-')

    def __mul__(self, other: Any) -> "Spectrum[T]":
        if isinstance(other, float):
            for comp in self.values:
                current_value, unit = comp.value, comp.default_unit
                comp.value = (current_value * other, unit)
            return self
        elif isinstance(other, Spectrum):
            return self._linear_op(other, '*')
        else:
            raise NotImplementedError


    def __truediv__(self, factor: float) -> "Spectrum[T]":
        return self * (1 / factor)

    def __repr__(self) -> str:
        lines = [f"{f}: {g}" for f, g in self.spectrum.items()]
        return f"{self.value_cls.__name__} Spectrum:\n  " + "\n  ".join(lines)

    def __iter__(self) -> Iterator[tuple[ct.Frequency, T]]:
        """
        Iterate over (frequency, value) pairs sorted by frequency.
        """
        return iter(sorted(self.spectrum.items(), key=lambda item: item[0]))

    def __getitem__(self, freq: ct.Frequency) -> T:
        if freq in self.frequencies:
            return self.spectrum[freq]
        else:
            raise ValueError(f"{freq} is not inside {self.__class__.__name__}")

    def __setitem__(self, freq: ct.Frequency, new_val: T) -> None:
        if freq in self.frequencies:
            self.spectrum[freq] = new_val
        else:
            raise ValueError(f"{freq} is not inside {self.__class__.__name__}")

    @property
    def mean(self) -> float:
        """
        Mean property retuns the mean value over the spectrum
        """
        mean: float = 0.0
        for v in self.spectrum.values():
            mean += v.value ** 2
        return mean**0.5/len(self.spectrum.values())

    def add_val(self, frequency: ct.Frequency, value: T, do_round: bool = True) -> None:
        """
        Adds a Frequency - value pair to the Spectrum dict
        """
        if do_round:
            frequency.Hz = round(frequency.Hz)
        self.spectrum[frequency] = value

    @property
    def frequencies(self):
        """"
        Property that returns the Frequencies in a spectrum as a list
        """
        return list(self.spectrum.keys())

    @property
    def values(self):
        """
        Property that retuns the spectrum values as a list
        """
        return list(self.spectrum.values())

    def peak_frequency(self) -> ct.Frequency:
        """Return the frequency with the maximum value in the spectrum."""
        if not self.spectrum:
            raise ValueError("Spectrum is empty")
        return max(self.spectrum.items(), key=lambda item: item[1].value)[0]

def mse(current: Spectrum[T], target: Spectrum[T]) -> float:
    return abs((current - target).mean)

def integral(spec: Spectrum[T]) -> float:
    sum_value = 0
    for comp in spec.values:
        sum_value += comp.value
    return sum_value
