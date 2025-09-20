from .units import Unit


class Frequency(Unit):
    default_unit = 'Hz'

    def __init__(self, value: float, unit: str):
        super().__init__(value=value, unit=unit)

    @property
    def Hz(self):
        return self.value

    @Hz.setter
    def Hz(self, new: float):
        self.value = (new, 'Hz')

    @property
    def MHz(self):
        return self._value/1e6

    @MHz.setter
    def MHz(self, new: float):
        self.value = (new, 'MHz')

    @property
    def THz(self):
        return self._value/1e12

    @THz.setter
    def THz(self, new: float):
        self.value = (new, 'THz')


