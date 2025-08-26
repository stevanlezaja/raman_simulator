from .units import Unit


class Frequency(Unit):
    default_unit = 'Hz'

    def __init__(self, value: float, unit: str):
        super().__init__(value=value, unit=unit)

    @property
    def Hz(self):
        return self._value

    @Hz.setter
    def Hz(self, new):
        self.value = (new, 'Hz')

    @property
    def MHz(self):
        return self._value

    @MHz.setter
    def MHz(self, new):
        self.value = (new, 'MHz')

    @property
    def THz(self):
        return self._value

    @THz.setter
    def THz(self, new):
        self.value = (new, 'THz')


