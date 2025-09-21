from typing import TypeVar, Generic, Type, Iterator, Optional

import custom_types as ct


class RamanInputs:
    def __init__(self, powers: Optional[list[ct.Power]] = None, wavelengths: Optional[list[ct.Length]] = None, n_pumps: Optional[int] = None):
        if powers is not None or wavelengths is not None:
            assert powers is not None and wavelengths is not None, "Both powers and wavelengths need to be provided"

        if powers is None:
            assert n_pumps is not None, "n_pumps cannot be None if powers are None"
            powers = [ct.Power(0.0, 'W') for _ in range(n_pumps)]

        if wavelengths is None:
            assert n_pumps is not None, "n_pumps cannot be None if wavelengths are None"
            wavelengths = [ct.Length(0.0, 'm') for _ in range(n_pumps)]

        self.wavelengths: list[ct.Length] = wavelengths
        self.powers: list[ct.Power] = powers


T = TypeVar("T", bound = ct.UnitProtocol)

class Spectrum(Generic[T]):

    operations = ['+', '-']

    def __init__(self, value_cls: Type[T]):
        self.value_cls: Type[T] = value_cls  # store class type
        self.spectrum: dict[ct.Frequency, T] = {}

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

    def add_val(self, frequency: ct.Frequency, value: T) -> None:
        self.spectrum[frequency] = value

    def __iter__(self) -> Iterator[tuple[ct.Frequency, T]]:
        return iter(self.spectrum.items())

    @property
    def frequencies(self):
        return list(self.spectrum.keys())

    @property
    def values(self):
        return list(self.spectrum.values())


if __name__ == "__main__":
    spec1 = Spectrum(ct.PowerGain)
    spec1.spectrum = {
        ct.Frequency(0, 'Hz'): ct.PowerGain(10, ''),
        ct.Frequency(1, 'Hz'): ct.PowerGain(15, ''),
        ct.Frequency(2, 'Hz'): ct.PowerGain(20, ''),
        ct.Frequency(3, 'Hz'): ct.PowerGain(25, ''),
        ct.Frequency(4, 'Hz'): ct.PowerGain(7, ''),
    }

    spec2 = Spectrum(ct.PowerGain)
    spec2.spectrum = {
        ct.Frequency(0, 'Hz'): ct.PowerGain(1, ''),
        ct.Frequency(1, 'Hz'): ct.PowerGain(2, ''),
        ct.Frequency(2, 'Hz'): ct.PowerGain(3, ''),
        ct.Frequency(3, 'Hz'): ct.PowerGain(4, ''),
        ct.Frequency(4, 'Hz'): ct.PowerGain(1, ''),
    }

    print(spec1 + spec2)