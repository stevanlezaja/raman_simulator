from abc import ABC, abstractmethod

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

    def split_exponent(token: str) -> tuple[str, int]:
        token_list = token.split('^')
        if len(token_list) == 2:
            token, exponenent = token_list[0], token_list[1]
        elif len(token_list) == 1:
            token = token_list[0]
            exponenent = 1
        else:
            raise Exception(f"Invalid expression {token}")

        return (token, exponenent)

    def split_unit(token: str) -> tuple[str, str]:
        for base in sorted(UnitRegistry.all_units(), key=len):
            if token.endswith(base):
                mul = token.removesuffix(base)
                return mul, base
        raise Exception(f"Unknown unit token: {token}")
    
    def process_unit(token: str) -> tuple[str, str, str]:
        token, exponent = Unit.split_exponent(token)
        mul, base = Unit.split_unit(token)
        return mul, base, exponent

    def __init__(self, *, value: float, unit: str):
        self._value = self.convert(value=value, unit=unit)

    def convert(self, *, value: float, unit: str):
        parts = unit.split("/")
        numerator = parts[0].split("*")

        multiplier = 1.0

        for num in numerator:
            mul, base, exp = Unit.process_unit(num)
            multiplier *= (Multipliers.units.get(mul, 1.0) ** int(exp))

        if len(parts) > 1:
            denominator = parts[1:]
            for den in denominator:
                mul, base, exp = Unit.process_unit(den)
                multiplier /= (Multipliers.units.get(mul, 1.0) ** exp)

        return (value * multiplier)

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
