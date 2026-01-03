from typing import TypeVar, Generic, Type, Iterator, Any
import numpy as np

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
    norm_min: float | None = None
    norm_max: float | None = None

    def __init__(self, value_cls: Type[T],* , frequencies: list[ct.Frequency] = [], values: list[ct.Power|ct.PowerGain] = [], normalized: bool = False):
        self.value_cls: Type[T] = value_cls
        assert len(frequencies) == len(values)
        self.spectrum: dict[ct.Frequency, T] = {}
        self.normalized = normalized
        for freq, val in zip(frequencies, values):
            assert isinstance(val, self.value_cls)
            self.add_val(freq, val)

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
                current_value, unit = comp.value, comp.default_unit  # type: ignore
                comp.value = (current_value * other, unit)  # type: ignore
            return self
        elif isinstance(other, Spectrum):
            return self._linear_op(other, '*')  # type: ignore
        else:
            raise NotImplementedError

    def __truediv__(self, other: Any) -> "Spectrum[Any]":
        if isinstance(other, float):
            return self * (1 / other)
        elif isinstance(other, Spectrum):
            return self._spectrum_div(other)  # type: ignore
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

    def as_array(self, include_freq: bool = True) -> np.ndarray:
        """
        Convert Spectrum into a flat array:
            [frequencies_Hz..., values...]
        Values use the appropriate unit:
            Power → W
            PowerGain → dB
        """
        freqs = [f.Hz for f in self.frequencies]

        if self.value_cls == ct.Power:
            vals = [v.W for v in self.values]  # type: ignore
        elif self.value_cls == ct.PowerGain:
            vals = [v.dB for v in self.values]  # type: ignore
        else:
            raise TypeError(f"Unsupported Spectrum value type: {self.value_cls}")

        arr = np.array(freqs + vals, dtype=float) if include_freq else np.array(vals, dtype=float)

        return arr

    @classmethod
    def from_array(cls, value_cls: Type[T], arr: np.ndarray) -> "Spectrum[T]":
        """
        Reconstruct Spectrum from a flat array:
            [f1, f2, ..., fN, v1, v2, ..., vN]
        Frequencies in Hz, values depend on value_cls.

        Example:
            Spectrum.from_array(ct.Power, arr)
        """

        arr = np.asarray(arr, dtype=float)
        n = len(arr)

        assert n % 2 == 0, "Array must contain N freqs + N values"

        half = n // 2
        freq_part = arr[:half]
        val_part  = arr[half:]

        frequencies = [ct.Frequency(float(f), "Hz") for f in freq_part]

        if value_cls == ct.Power:
            values = [ct.Power(float(v), "W") for v in val_part]
        elif value_cls == ct.PowerGain:
            values = [ct.PowerGain(float(v), "dB") for v in val_part]
        else:
            raise TypeError(f"Unsupported Spectrum value type: {value_cls}")

        return cls(value_cls, frequencies=frequencies, values=values)  # type: ignore


    def peak_frequency(self) -> ct.Frequency:
        """Return the frequency with the maximum value in the spectrum."""
        if not self.spectrum:
            raise ValueError("Spectrum is empty")
        return max(self.spectrum.items(), key=lambda item: item[1].value)[0]

    def to_dict(self) -> dict[str, Any]:
        result = {"frequencies_Hz": [f.Hz for f in self.frequencies],}
        if self.value_cls == ct.Power:
            result["values_mW"] = [v.mW for v in self.values]  # type: ignore
        elif self.value_cls == ct.PowerGain:
            result['values_dB'] = [v.dB for v in self.values]  # type: ignore
        else:
            raise TypeError("Unsupported Spectrum value type")
        return result

    @classmethod
    def from_dict(cls, data: dict):  # type: ignore
        freqs = [ct.Frequency(float(f), 'Hz') for f in data['frequencies_Hz']]  # type: ignore
        if 'values_mW' in data.keys():
            vals = [ct.Power(float(v), 'mW') for v in data["values_mW"]]  # type: ignore
            cls_type = ct.Power
        elif 'values_dB' in data.keys():
            vals = [ct.PowerGain(float(v), 'dB') for v in data["values_dB"]]  # type: ignore
            cls_type = ct.PowerGain
        else:
            raise TypeError("Unsupported Spectrum value type")
        return cls(cls_type, frequencies=freqs, values=vals)  # type: ignore

    @classmethod
    def set_normalization_limits(cls, min_val: float, max_val: float):
        cls.norm_min = min_val
        cls.norm_max = max_val

    def normalize(self) -> "Spectrum[T]":
        """
        Normalize spectrum values in-place to [0, 1] using the provided limits.
        """
        assert not self.normalized
        assert isinstance(Spectrum.norm_max, float) and isinstance(Spectrum.norm_min, float), f"Norm max is {type(Spectrum.norm_max)}, Norm min is {type(Spectrum.norm_min)}"
        denom = Spectrum.norm_max - Spectrum.norm_min
        if denom == 0:
            raise ValueError("Cannot normalize when min_val == max_val")

        for f, v in self.spectrum.items():
            x = (v.value - Spectrum.norm_min) / denom
            self.spectrum[f] = self.value_cls(x, v.default_unit)  # type: ignore
        self.normalized = True
        return self

    def denormalize(self) -> "Spectrum[T]":
        """
        Denormalize spectrum values in-place using the provided limits.
        """
        assert self.normalized
        assert isinstance(Spectrum.norm_max, float) and isinstance(Spectrum.norm_min, float)
        denom = Spectrum.norm_max - Spectrum.norm_min
        if denom == 0:
            raise ValueError("Cannot denormalize when min_val == max_val")

        for f, v in self.spectrum.items():
            x = v.value * denom + Spectrum.norm_min
            self.spectrum[f] = self.value_cls(x, v.default_unit)  # type: ignore
        self.normalized = False
        return self


def mse(current: Spectrum[T], target: Spectrum[T]) -> float:
    return abs((current - target).mean)

def integral(spec: Spectrum[T]) -> ct.Power:
    if spec.value_cls == ct.Power:
        sum_value = ct.Power(0, 'W')
        for comp in spec.values:
            assert isinstance(comp, ct.Power)
            sum_value += comp
        return sum_value
    raise NotImplementedError
