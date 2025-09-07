from .units import Unit, UnitRegistry

class Length(Unit):
    default_unit = 'm'

    def __init__(self, value: float, unit: str):
        super().__init__(value=value, unit=unit)

    @property
    def nm(self):
        return self._value * 1e9

    @nm.setter
    def nm(self, new):
        self.value = (new, 'nm')

    @property
    def m(self):
        return self._value

    @m.setter
    def m(self, new):
        self.value = (new, 'm')

    @property
    def km(self):
        return self._value / 1e3

    @km.setter
    def km(self, new):
        self.value = (new, 'km')


if __name__ == "__main__":
    distance = Length(value=10, unit='km')
    print(distance)
    distance.value = (15, 'mm')
    print(distance)