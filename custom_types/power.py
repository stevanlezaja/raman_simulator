from .units import Unit


class Power(Unit):
    default_unit = 'W'

    def __init__(self, value: float, unit: str):
        super().__init__(value=value, unit=unit)

    @property
    def W(self):
        return self._value

    @W.setter
    def W(self, new):
        self.value = (new, 'W')

    @property
    def mW(self):
        return self._value

    @mW.setter
    def mW(self, new):
        self.value = (new, 'mW')

