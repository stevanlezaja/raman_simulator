class UnitRegistry:
    base_units: set[str] = set()

    @classmethod
    def register(cls, unit: str):
        cls.base_units.add(unit)

    @classmethod
    def all_units(cls):
        return sorted(cls.base_units, key=len)

