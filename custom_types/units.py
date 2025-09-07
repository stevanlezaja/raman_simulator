from abc import ABC, abstractmethod
from typing import Optional

from .unit_registry import UnitRegistry


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
        degree = 1
        if hasattr(self, 'degree'): degree *= self.degree
        return value * Multipliers.units[mul] ** degree

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new):
        new_value, new_unit = new
        self._value = self.convert(value=new_value, unit=new_unit)

    def __init_subclass__(cls):
        super().__init_subclass__()
        if not any(c in cls.default_unit for c in ['/', '*', '^']):
            UnitRegistry.register(cls.default_unit, cls.__name__)

    def __str__(self):
        return f"{self._value} {self.default_unit}"
    
    def __hash__(self):
        return hash(round(self.value, 12))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, self.__class__):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, self.__class__):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, self.__class__):
            return self.value >= other.value
        return NotImplemented
