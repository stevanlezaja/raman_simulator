
from typing import TypeVar, Generic, Type, Iterator, Any

import custom_types as ct
import custom_logging as clog


log = clog.get_logger("Spectrum")


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
        if isinstance(other, float | int):
            for comp in self.values:
                current_value, unit = comp.value, comp.default_unit
                comp.value = (current_value * other, unit)
            return self
        elif isinstance(other, Spectrum):
            return self._linear_op(other, '*')
        else:
            raise NotImplementedError

    def __truediv__(self, other: Any) -> "Spectrum[Any]":
        if isinstance(other, float):
            return self * (1 / other)
        elif isinstance(other, Spectrum):
            return self._spectrum_div(other)
        else:
            raise NotImplementedError

    def _spectrum_div(self: "Spectrum[ct.Power]", other: "Spectrum[ct.Power]") -> "Spectrum[ct.PowerGain]":

        if list(self.spectrum.keys()) != list(other.spectrum.keys()):
            raise ValueError("Spectrum division requires matching frequency samples.")

        result = Spectrum(ct.PowerGain)

        for f in self.spectrum:
            p_out = self.spectrum[f].value
            p_in  = other.spectrum[f].value

            gain_linear = p_out / p_in
            gain = ct.PowerGain(gain_linear, ' ')

            result.spectrum[f] = gain

        return result

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
