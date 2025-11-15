import numpy as np

from .units import Unit
from .conversions import db_to_linear_power


class PowerGain(Unit):
    default_unit = ' '

    def __init__(self, value: float = 0.0, unit: str = ' '):
        if unit == 'dB':
            value = db_to_linear_power(value)
            unit = ' '
        super().__init__(value=value, unit=unit)

    @property
    def dB(self):
        return 10 * np.log10(self.value)

    @dB.setter
    def dB(self, new: float):
        self.value = (10 ** (new / 10), ' ')

    @property
    def linear(self):
        return self.value

    @linear.setter
    def linear(self, new: float):
        self.value = (new, ' ')


if __name__ == "__main__":
    g = PowerGain()
    g.linear = 2
    assert int(g.dB) == 3
    g.dB = 10
    assert g.linear == 10
    g.dB = 3
    assert int(abs(g.linear - 2)) == 0
