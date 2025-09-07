from .units import Unit
from .unit_registry import UnitRegistry


class FiberAttenuation(Unit):
    default_unit = '1/m'

    def __init__(self, value: float, unit: str):
        super().__init__(value=value, unit=unit)


if __name__ == "__main__":
    print(UnitRegistry.all_units())
    att = FiberAttenuation(0.1, '1/km')
    print(att)