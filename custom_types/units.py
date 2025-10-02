from abc import ABC
from typing import TypeVar, Protocol, Self

import custom_logging as clog

from .unit_registry import UnitRegistry


log = clog.get_logger("Unit")


class UnitProtocol(Protocol):
    value: float
    def __add__(self, other: Self) -> Self: ...
    def __sub__(self, other: Self) -> Self: ...


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


class Unit(ABC, UnitProtocol):
    allow_negative = True
    default_unit = ''
    T = TypeVar("T", bound="Unit")

    @staticmethod
    def split_exponent(token: str) -> tuple[str, int]:
        token_list = token.split('^')
        if len(token_list) == 2:
            token, exponenent = token_list[0], int(token_list[1])
        elif len(token_list) == 1:
            token = token_list[0]
            exponenent = 1
        else:
            raise Exception(f"Invalid expression {token}")

        return (token, exponenent)

    @staticmethod
    def split_unit(token: str) -> tuple[str, str]:
        for base in sorted(UnitRegistry.all_units(), key=len):
            if token.endswith(base):
                mul = token.removesuffix(base)
                return mul, base
        raise Exception(f"Unknown unit token: {token}")
    
    @staticmethod
    def process_unit(token: str) -> tuple[str, str, int]:
        token, exponent = Unit.split_exponent(token)
        mul, base = Unit.split_unit(token)
        return mul, base, exponent

    def __init__(self, *, value: float, unit: str):
        self._value = self.convert(value=value, unit=unit)

    def convert(self, *, value: float, unit: str) -> float:
        parts = unit.split("/")
        numerator = parts[0].split("*")

        multiplier = 1.0

        for num in numerator:
            mul, _, exp = Unit.process_unit(num)
            multiplier *= (Multipliers.units.get(mul, 1.0) ** int(exp))

        if len(parts) > 1:
            denominator = parts[1:]
            for den in denominator:
                mul, _, exp = Unit.process_unit(den)
                multiplier /= (Multipliers.units.get(mul, 1.0) ** exp)

        return (value * multiplier)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new: tuple[float, str]): # type: ignore
        new_value, new_unit = new
        if not self.allow_negative:
            if new_value < 0:
                new_value = 0.0
        self._value = self.convert(value=new_value, unit=new_unit)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if cls.__name__ in UnitRegistry.base_units.keys():
            log.info("Unit %s already registered", cls.__name__)
            return
        if not any(c in cls.default_unit for c in ['/', '*', '^']):
            log.info("Registering %s: Default unit: %s", cls.__name__, cls.default_unit)
            UnitRegistry.register(cls.default_unit, cls.__name__)
            return
        log.info("%s unit %s is not a base unit. Not registering", cls.__name__, cls.default_unit)

    def __str__(self):
        return f"{self._value} {self.default_unit}"

    def __repr__(self):
        return f"{self._value} {self.default_unit}"
    
    def __hash__(self):
        return hash(round(self.value, 12))

    def __eq__(self, other: object):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return NotImplemented

    def __lt__(self, other: object):
        if isinstance(other, self.__class__):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other: object):
        if isinstance(other, self.__class__):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other: object):
        if isinstance(other, self.__class__):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other: object):
        if isinstance(other, self.__class__):
            return self.value >= other.value
        return NotImplemented

    def __add__(self: T, other: T) -> T:
        assert isinstance(other, self.__class__), (f"Both operands need to be type {self.__class__.__name__}")
        result = self.__class__(value=self.value, unit=self.default_unit)
        result.value = (self.value + other.value, self.default_unit)
        return result

    def __sub__(self: T, other: T) -> T:
        assert isinstance(other, self.__class__), (f"Both operands need to be type {self.__class__.__name__}")
        result = self.__class__(value=self.value, unit=self.default_unit)
        result.value = (self.value - other.value, self.default_unit)
        return result

    def __neg__(self: T) -> T:
        self._value = -self._value
        return self
