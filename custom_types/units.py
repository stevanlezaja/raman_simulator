from abc import ABC, abstractmethod


class Multipliers:
    units = {
        "p": 1e-12,
        "n": 1e-9,
        "u": 1e-6,
        "m": 1e-3,
        "": 1.0,
        "k": 1e3,
        "M": 1e6,
        "G": 1e9,
        "T": 1e12,
    }


class Unit(ABC):
    default_unit = ''
    
    def __init__(self, *, value: float, unit: str):
        self._value = self.convert(value=value, unit=unit)

    def convert(self, *, value: float, unit: str):
        mul = unit.removesuffix(self.__class__.default_unit)
        return value * Multipliers.units[mul]

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, new):
        new_value, new_unit = new
        self._value = self.convert(value=new_value, unit=new_unit)

    def __str__(self):
        return f"{self._value} {self.default_unit}"

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