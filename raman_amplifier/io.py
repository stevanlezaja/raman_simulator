from typing import TypeVar, Generic, Type

from custom_types import PowerGain, Length, Power, Frequency, UnitProtocol


class RamanInputs:
    def __init__(self):
        self.wavelengths: list[Length] = []
        self.powers: list[Power] = []


T = TypeVar("T", bound = UnitProtocol)

class Spectrum(Generic[T]):

    operations = ['+', '-']

    def __init__(self, value_cls: Type[T]):
        self.value_cls: Type[T] = value_cls  # store class type
        self.spectrum: dict[Frequency, T] = {}

    def _linear_op(self, other: "Spectrum[T]", operation: str) -> "Spectrum[T]":
        assert operation in Spectrum.operations, f"Operation {operation} is not supported"
        assert isinstance(other, Spectrum), (f"Both operands need to be type {self.__class__.__name__}")
        assert self.spectrum.keys() == other.spectrum.keys(), ("Both spectra need to have the same frequencies")
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

    @property
    def mean(self) -> float:
        mean: float = 0.0
        for v in self.spectrum.values():
            mean += v.value
        return mean/len(self.spectrum.values())

    def add_val(self, frequency: Frequency, value: T) -> None:
        self.spectrum[frequency] = value


if __name__ == "__main__":
    spec1 = Spectrum(PowerGain)
    spec1.spectrum = {
        Frequency(0, 'Hz'): PowerGain(10, ''),
        Frequency(1, 'Hz'): PowerGain(15, ''),
        Frequency(2, 'Hz'): PowerGain(20, ''),
        Frequency(3, 'Hz'): PowerGain(25, ''),
        Frequency(4, 'Hz'): PowerGain(7, ''),
    }

    spec2 = Spectrum(PowerGain)
    spec2.spectrum = {
        Frequency(0, 'Hz'): PowerGain(1, ''),
        Frequency(1, 'Hz'): PowerGain(2, ''),
        Frequency(2, 'Hz'): PowerGain(3, ''),
        Frequency(3, 'Hz'): PowerGain(4, ''),
        Frequency(4, 'Hz'): PowerGain(1, ''),
    }

    print(spec1 + spec2)