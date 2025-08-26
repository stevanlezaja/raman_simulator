from .units import Unit

class Length(Unit):
    default_unit = 'm'

    def __init__(self, value: float, unit: str):
        super().__init__(value=value, unit=unit)

    @property
    def m(self):
        return self._value

    @property
    def km(self):
        return self._value/1e-3


if __name__ == "__main__":
    distance = Length(value=10, unit='km')
    print(distance)
    distance.value = (15, 'mm')
    print(distance)