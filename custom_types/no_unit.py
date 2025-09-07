from .units import Unit
from .unit_registry import UnitRegistry


class NoUnit(Unit):
    default_unit = '1'

    def __init__(self):
        super().__init__(value=1.0, unit=self.default_unit)

if __name__ == "__main__":
    print(UnitRegistry.all_units())
    nounit = NoUnit()
    print(nounit)