from abc import ABC, abstractmethod
import numpy as np

from custom_types import Length, Area


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
        freq = np.array(sorted(eff_dict.keys()))
        eff = np.array([eff_dict[k] for k in freq])
        return np.interp(delta_f, freq, eff)

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


class DispersionCompensatingFiber(Fiber):
    @property
    def raman_efficiency(self) -> dict:
        return {
            0: 0,
            5: 0.1,
            10: 1.5,
            15: 2.8,
            20: 0.5
        }
    
    @property
    def effective_area(self):
        return Area(15, 'um^2')


class NonZeroDispersionFiber(Fiber):
    @property
    def raman_efficiency(self) -> dict:
        return {
            0: 0,
            5: 0.25,
            10: 0.4,
            15: 0.5,
            20: 0.1
        }
    
    @property
    def effective_area(self):
        return Area(55, 'um^2')


class SuperLargeEffectiveArea(Fiber):
    @property
    def raman_efficiency(self) -> dict:
        return {
            0: 0,
            5: 0.1,
            10: 0.15,
            15: 0.25,
            20: 0.05
        }
    
    @property
    def effective_area(self):
        return Area(105, 'um^2')