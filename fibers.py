from abc import ABC, abstractmethod


class Fiber(ABC):
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
    def effective_area(self) -> float:
        """"
            float
              - effective area of fiber [um^2]
        """
        return 0.0


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
        return 15 * 1e-12


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
        return 55 * 1e-12


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
        return 105 * 1e-12