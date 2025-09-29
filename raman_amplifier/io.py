"""
This module contains Input and Output types for a Raman Amplifier
"""

from typing import TypeVar, Generic, Type, Iterator, Optional

import custom_types as ct
import custom_logging as clog


log = clog.get_logger("IO")


class RamanInputs:
    """
    RamanInputs is a class used to represent inputs to a Raman Amplifier
        It contains the Wavelength - Power pairs representing the Pump state
    """
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

    def __repr__(self) -> str:
        lines = [f"{f}: {g}" for f, g in self.spectrum.items()]
        return "Gain Spectrum:\n  " + "\n  ".join(lines)

    def __iter__(self) -> Iterator[tuple[ct.Frequency, T]]:
        """
        Iterate over (frequency, value) pairs sorted by frequency.
        """
        return iter(sorted(self.spectrum.items(), key=lambda item: item[0]))

    def __getitem__(self, freq: ct.Frequency) -> T:
        if freq in self.frequencies:
            return self.spectrum[freq]
        raise ValueError(f"{freq} is not inside {self.__class__.__name__}")

    def __setitem__(self, freq: ct.Frequency, new_val: T) -> None:
        if freq in self.frequencies:
            self.spectrum[freq] = new_val
        raise ValueError(f"{freq} is not inside {self.__class__.__name__}")

    @property
    def mean(self) -> float:
        """
        Mean property retuns the mean value over the spectrum
        """
        mean: float = 0.0
        for v in self.spectrum.values():
            mean += v.value
        return mean/len(self.spectrum.values())

    def add_val(self, frequency: ct.Frequency, value: T) -> None:
        """
        Adds a Frequency - value pair to the Spectrum dict
        """
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
