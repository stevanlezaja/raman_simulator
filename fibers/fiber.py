"""
Module defining various types of optical fibers and their Raman gain characteristics.

This module provides base and specific fiber classes with properties such as
length, effective area, signal/pump loss, and Raman gain efficiency. It also
includes a method for plotting Raman efficiency.
"""

from abc import ABC, abstractmethod
import csv
import re

import numpy as np
from numpy.typing import ArrayLike
from matplotlib.axes import Axes

import custom_types as ct


class NegativeLength(Exception):
    """
    Exception raised when a fiber is initialized with a negative length.

    Attributes:
        length (Length): The invalid fiber length.
        msg (str): Explanation of the error.
    """
    def __init__(self, length: ct.Length, msg: str = "Fiber length must be non-negative!"):
        """
        Initialize the exception.

        Args:
            length (Length): The negative length that caused the error.
            msg (str, optional): Error message. Defaults to a standard message.
        """
        super().__init__(f"{msg}: {length}")


class Fiber(ABC):
    """
    Abstract base class for optical fibers.

    Defines common fiber attributes such as length, signal/pump attenuation,
    and requires subclasses to implement Raman efficiency and effective area.
    """
    def __init__(self):
        """Initialize default fiber parameters."""
        self.length = ct.Length(25.0, 'km')
        self.__alpha_p = ct.FiberAttenuation(0.0437, '1/km')
        self.__alpha_s = ct.FiberAttenuation(0.0576, '1/km')

    @property
    def name(self) -> str:
        """
        Generate a human-readable fiber name from the class name.

        Returns:
            str: Fiber name in spaced PascalCase.
        """
        pascal_case = self.__class__.__name__
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', pascal_case)
        return words

    def C_R(self, delta_f: ct.Frequency) -> float:
        """
        Interpolate Raman gain efficiency at a given pump-signal frequency difference.

        Args:
            delta_f (Frequency): Frequency difference between pump and signal.

        Returns:
            float: Raman gain efficiency [1/(W*km)].
        """
        eff_dict = self.raman_efficiency
        freq = eff_dict.keys()
        eff = np.array([eff_dict[k] for k in freq])
        return float(np.interp(delta_f.Hz, [f.Hz for f in freq], eff))

    @property
    def alpha_s(self) -> ct.FiberAttenuation:
        """
        Fiber attenuation at the signal frequency.

        Returns:
            FiberAttenuation: Signal attenuation [1/km].
        """
        return self.__alpha_s

    @property
    def alpha_p(self):
        """
        Fiber loss at pump frequency

        Returns:
            FiberAttenuation: Signal attenuation [1/km].
        """
        return self.__alpha_p

    @property
    @abstractmethod
    def raman_efficiency(self) -> dict[ct.Frequency, float]:
        """
        Raman gain efficiency spectrum of the fiber.

        Returns:
            dict[Frequency, float]: Mapping of pump-signal frequency difference
            [Hz] to Raman gain efficiency [1/(W*km)].
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def effective_area(self) -> ct.Area:
        """
        Effective core area of the fiber.

        Returns:
            Area: Effective area in square meters or square micrometers.
        """
        raise NotImplementedError()

    def plot_raman_efficiency(self, ax: Axes, x_points: int=100) -> Axes:
        """
        Plot the Raman efficiency spectrum of the fiber.

        Args:
            ax (Axes): Matplotlib Axes object to plot on.
            x_points (int, optional): Number of points for the frequency axis.
                Defaults to 100.

        Returns:
            Axes: The same Axes object with the plot added.
        """
        max_freq = max(self.raman_efficiency.keys()).Hz

        x: ArrayLike = np.linspace(0, max_freq, x_points)
        y: list[float] = []

        for delta_f in x:
            y.append(self.C_R(ct.Frequency(delta_f, 'Hz')))

        plot_name = self.name
        ax.plot(x/1e12, y, label=plot_name) # type: ignore[call-overload]
        ax.set_title(plot_name) # type: ignore[call-overload]
        ax.set_xlabel("delta frequency [THz]") # type: ignore[call-overload]
        ax.set_ylabel("Raman efficiency [1/W/km]") # type: ignore[call-overload]
        ax.legend() # type: ignore[call-overload]

        return ax


class StandardSingleModeFiber(Fiber):
    """Standard single-mode fiber with measured Raman gain efficiency in the C-band."""

    @property
    def raman_efficiency(self) -> dict[ct.Frequency, float]:
        data: dict[ct.Frequency, float] = {}
        with open(
            "data/Raman_Gain_efficiency_SSMF_C-band_September2025.csv",
            "r",
            encoding="utf-8"
        ) as f:
            reader = csv.reader(f)
            next(reader)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    freq = ct.Frequency(abs(float(row[0])), 'THz')
                    value = float(row[1])
                    data[freq] = value
        return data

    @property
    def effective_area(self) -> ct.Area:
        return ct.Area(80, 'um^2')



class DispersionCompensatingFiber(Fiber):
    """Fiber with small effective area designed for dispersion compensation."""

    @property
    def raman_efficiency(self) -> dict[ct.Frequency, float]:
        return {
            ct.Frequency(0, 'THz'): 0,
            ct.Frequency(5, 'THz'): 0.1,
            ct.Frequency(10, 'THz'): 1.5,
            ct.Frequency(15, 'THz'): 2.8,
            ct.Frequency(20, 'THz'): 0.5
        }

    @property
    def effective_area(self) -> ct.Area:
        return ct.Area(15, 'um^2')


class NonZeroDispersionFiber(Fiber):
    """Fiber with moderate effective area and non-zero dispersion."""

    @property
    def raman_efficiency(self) -> dict[ct.Frequency, float]:
        return {
            ct.Frequency(0, 'THz'): 0,
            ct.Frequency(5, 'THz'): 0.25,
            ct.Frequency(10, 'THz'): 0.4,
            ct.Frequency(15, 'THz'): 0.5,
            ct.Frequency(20, 'THz'): 0.1
        }

    @property
    def effective_area(self) -> ct.Area:
        return ct.Area(55, 'um^2')


class SuperLargeEffectiveArea(Fiber):
    """Fiber with very large effective area."""

    @property
    def raman_efficiency(self) -> dict[ct.Frequency, float]:
        return {
            ct.Frequency(0, 'THz'): 0,
            ct.Frequency(5, 'THz'): 0.1,
            ct.Frequency(10, 'THz'): 0.15,
            ct.Frequency(15, 'THz'): 0.25,
            ct.Frequency(20, 'THz'): 0.05
        }

    @property
    def effective_area(self) -> ct.Area:
        return ct.Area(105, 'um^2')

class ChristopheExperimentFiber(Fiber):
    """Fiber representing experimental parameters from Christophe's measurements."""

    @property
    def raman_efficiency(self):
        return{
            ct.Frequency(0, 'THz'): 0.42,
            ct.Frequency(25, 'THz'): 0.42,
        }

    @property
    def effective_area(self) -> ct.Area:
        return ct.Area(80, 'um^2')
