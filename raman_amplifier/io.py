from custom_types import PowerGain, Length, Power, Frequency


class RamanInputs:
    def __init__(self):
        self.wavelengths = list[Length]
        self.powers = list[Power]


class GainSpectrum:

    operations = ['+', '-']

    def __init__(self):
        self.spectrum: dict[Frequency, PowerGain] = {}

    def _linear_op(self, other: "GainSpectrum", operation: str):
        assert operation in GainSpectrum.operations, f"Operation {operation} is not supported"
        assert isinstance(other, GainSpectrum), (f"Both operands need to be type {self.__class__.__name__}")
        assert self.spectrum.keys() == other.spectrum.keys(), ("Both spectra need to have the same frequencies")
        result = GainSpectrum()
        for f, g in self.spectrum.items():
            result.spectrum[f] = g + other.spectrum[f]
        return result

    def __add__(self, other: "GainSpectrum") -> "GainSpectrum":
        return self._linear_op(other, '+')

    def __sub__(self, other: "GainSpectrum") -> "GainSpectrum":
        return self._linear_op(other, '-')

    def __repr__(self) -> str:
        lines = [f"{f}: {g}" for f, g in self.spectrum.items()]
        return "Gain Spectrum:\n  " + "\n  ".join(lines)


if __name__ == "__main__":
    spec1 = GainSpectrum()
    spec1.spectrum = {
        Frequency(0, 'Hz'): PowerGain(10, ''),
        Frequency(1, 'Hz'): PowerGain(15, ''),
        Frequency(2, 'Hz'): PowerGain(20, ''),
        Frequency(3, 'Hz'): PowerGain(25, ''),
        Frequency(4, 'Hz'): PowerGain(7, ''),
    }

    spec2 = GainSpectrum()
    spec2.spectrum = {
        Frequency(0, 'Hz'): PowerGain(1, ''),
        Frequency(1, 'Hz'): PowerGain(2, ''),
        Frequency(2, 'Hz'): PowerGain(3, ''),
        Frequency(3, 'Hz'): PowerGain(4, ''),
        Frequency(4, 'Hz'): PowerGain(1, ''),
    }

    print(spec1 + spec2)