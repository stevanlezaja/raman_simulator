class UnitRegistry:
    base_units: dict[str, str] = dict()

    @classmethod
    def register(cls, unit: str, class_name: str):
        cls.base_units[class_name] = unit

    @classmethod
    def all_units(cls):
        return cls.base_units.values()

