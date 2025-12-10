"""
This module contains Input and Output types for a Raman Amplifier
"""

from typing import Optional, Any
import numpy as np

import custom_types as ct
import custom_logging as clog


log = clog.get_logger("IO")


class RamanInputs:
    """
    RamanInputs is a class used to represent inputs to a Raman Amplifier
        It contains the Wavelength - Power pairs representing the Pump state
    """

    MAX_POWER_W = 0.99
    MIN_POWER_W = 0.0

    MAX_WAVELENGTH_NM = 1480
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

    def as_array(self) -> np.ndarray:
        """
        Convert RamanInputs to a flat numpy array:
            [powers..., wavelengths...]
        Powers in W, wavelengths in nm.
        """

        power_values = [p.W for p in self.powers]
        wavelength_values = [wl.nm for wl in self.wavelengths]

        return np.array(power_values + wavelength_values, dtype=float)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "RamanInputs":
        """
        Reconstruct RamanInputs from a flat array:
            [P1, ..., Pn, λ1, ..., λn]
        """

        arr = np.asarray(arr, dtype=float)
        total = len(arr)
        assert total % 2 == 0, "Input array must have even length: N powers + N wavelengths"

        n_pumps = total // 2

        powers_vals = arr[:n_pumps]
        wavelengths_vals = arr[n_pumps:]

        powers = [ct.Power(float(p), 'W') for p in powers_vals]
        wavelengths = [ct.Length(float(wl), 'nm') for wl in wavelengths_vals]

        return cls(powers=powers, wavelengths=wavelengths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "powers_mW": [p.mW for p in self.powers],
            "wavelengths_nm": [w.nm for w in self.wavelengths],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        powers = [ct.Power(float(v), 'mW') for v in data["powers_mW"]]
        wavelengths = [ct.Length(float(w), 'nm') for w in data["wavelengths_nm"]]
        return cls(powers, wavelengths)

    def normalize(self) -> 'RamanInputs':
        """
        Normalize powers and wavelengths in-place to [0, 1] based on defined ranges.
        After normalization both powers[i] and wavelengths[i] become unitless
        ct.Power/ct.Length values whose .value is within [0, 1].
        """

        p_min, p_max = self.power_range
        wl_min, wl_max = self.wavelength_range
        power_span = p_max.value - p_min.value
        for i, p in enumerate(self.powers):
            norm_val = (p.value - p_min.value) / power_span
            self.powers[i] = ct.Power(norm_val, p.default_unit)
        wl_span = wl_max.value - wl_min.value
        for i, wl in enumerate(self.wavelengths):
            norm_val = (wl.value - wl_min.value) / wl_span
            self.wavelengths[i] = ct.Length(norm_val, wl.default_unit)
        self.value_dict = dict(zip(self.wavelengths, self.powers))

        return self

    def denormalize(self) -> 'RamanInputs':
        """
        Convert a normalized RamanInputs object (values in [0,1]) back into
        physical units based on defined ranges.
        """

        p_min, p_max = self.power_range
        wl_min, wl_max = self.wavelength_range
        power_span = p_max.value - p_min.value
        wl_span = wl_max.value - wl_min.value
        self.powers = [
            ct.Power(p.value * power_span + p_min.value, "W")
            for p in self.powers
        ]
        self.wavelengths = [
            ct.Length(w.value * wl_span + wl_min.value, 'm')
            for w in self.wavelengths
        ]

        return self

class RamanInputControl(RamanInputs):
    def denormalize(self) -> RamanInputs:
        """
        Convert a normalized RamanInputControl object (values in [0,1]) back into
        physical units based on defined ranges.
        """

        p_min, p_max = self.power_range
        wl_min, wl_max = self.wavelength_range
        power_span = p_max.value - p_min.value
        wl_span = wl_max.value - wl_min.value
        self.powers = [
            ct.Power(p.value * power_span, "W")
            for p in self.powers
        ]
        self.wavelengths = [
            ct.Length(w.value * wl_span, 'm')
            for w in self.wavelengths
        ]

        return self