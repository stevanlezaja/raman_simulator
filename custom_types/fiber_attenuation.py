import numpy as np

from .units import Unit


class FiberAttenuation(Unit):
    default_unit = '1/m'

    def __init__(self, value: float, unit: str):
        super().__init__(value=value, unit=unit)

    @property
    def km(self):
        return self.value * 1e3

    @km.setter
    def km(self, new):
        self.value = (new, '1/km')

    @property
    def m(self):
        return self.value

    @m.setter
    def m(self, new):
        self.value = (new, '1/m')

    @property
    def dB_km(self):
        return (10 / np.log(10)) * np.asarray(self.value)

    @dB_km.setter
    def dB_km(self, new):
        self.value = ((np.log(10) / 10) * np.asarray(new), '1/km')

if __name__ == "__main__":
    att = FiberAttenuation(0.1, '1/km')
    print(att)