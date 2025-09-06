from abc import ABC, abstractmethod
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from custom_types import Length, Area, Frequency


class NegativeLength(Exception):
    """Exception raised when fiber length is negative"""
    def __init__(self, length, msg="Fiber length must be non-negative!"):
        super().__init__(f"{msg}: {length}")


class Fiber(ABC):
    def __init__(self):
        self.length = Length(25.0, 'km')
        self.__alpha_s = 4.605 * 1e-5  # [m^-1]
        self.__alpha_p = 4.605 * 1e-5  # [m^-1]
    
    def C_R(self, delta_f):
        eff_dict = self.raman_efficiency
        freq = eff_dict.keys()
        eff = np.array([eff_dict[k] for k in freq])
        return np.interp(delta_f, [f.Hz for f in freq], eff)

    @property
    def alpha_s(self):
        """Fiber loss at signal frequency"""
        return self.__alpha_s

    @alpha_s.setter
    def alpha_s(self, value):
        self.__alpha_s = value

    @property
    def alpha_p(self):
        """Fiber loss at pump frequency"""
        return self.__alpha_p

    @alpha_p.setter
    def alpha_p(self, value):
        self.__alpha_p = value

    @property
    @abstractmethod
    def raman_efficiency(self) -> dict:
        """"
            dict:
              - keys: pump-signal frequency difference [THz]
              - values: raman gain efficiency [W^-1 km^-1]
        """
        return {}
    
    @property
    @abstractmethod
    def effective_area(self) -> Area:
        return Area(0.0, 'm')

    def make_raman_efficiency_figure(self, x_points: int=100) -> Figure:
        fig, ax = plt.subplots()

        x = np.linspace(0, max(self.raman_efficiency.keys()), x_points)
        y = []

        for delta_f in x:
            y.append(self.C_R(delta_f))

        plot_name = f"Raman Efficiency of {self.__class__.__name__}"
        ax.plot(x, y, label=plot_name)
        ax.set_title(plot_name)
        ax.set_xlabel("delta frequency")
        ax.set_ylabel("Raman efficiency")
        ax.legend()

        return fig


class StandardSingleModeFiber(Fiber):
    @property
    def raman_efficiency(self) -> dict:
        data = {}
        with open("data/Raman_Gain_efficiency_SSMF_C-band_September2025.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    freq = Frequency(abs(float(row[0])), 'THz')
                    value = float(row[1])
                    data[freq] = value
        return data

    @property
    def effective_area(self):
        raise NotImplementedError()



class DispersionCompensatingFiber(Fiber):
    @property
    def raman_efficiency(self) -> dict:
        return {
            Frequency(0, 'THz'): 0,
            Frequency(5, 'THz'): 0.1,
            Frequency(10, 'THz'): 1.5,
            Frequency(15, 'THz'): 2.8,
            Frequency(20, 'THz'): 0.5
        }
    
    @property
    def effective_area(self):
        return Area(15, 'um^2')


class NonZeroDispersionFiber(Fiber):
    @property
    def raman_efficiency(self) -> dict:
        return {
            Frequency(0, 'THz'): 0,
            Frequency(5, 'THz'): 0.25,
            Frequency(10, 'THz'): 0.4,
            Frequency(15, 'THz'): 0.5,
            Frequency(20, 'THz'): 0.1
        }
    
    @property
    def effective_area(self):
        return Area(55, 'um^2')


class SuperLargeEffectiveArea(Fiber):
    @property
    def raman_efficiency(self) -> dict:
        return {
            Frequency(0, 'THz'): 0,
            Frequency(5, 'THz'): 0.1,
            Frequency(10, 'THz'): 0.15,
            Frequency(15, 'THz'): 0.25,
            Frequency(20, 'THz'): 0.05
        }
    
    @property
    def effective_area(self):
        return Area(105, 'um^2')