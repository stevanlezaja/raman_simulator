from .units import Unit


class FiberAttenuation(Unit):
    default_unit = '1/m'

    def __init__(self, value: float, unit: str):
        super().__init__(value=value, unit=unit)

    @property
    def km(self):
        return self.value / 1e3

    @km.setter
    def km(self, new):
        self.value = (new * 1e3, 'km')

    @property
    def m(self):
        return self.value

    @m.setter
    def m(self, new):
        self.value = (new, 'm')


if __name__ == "__main__":
    att = FiberAttenuation(0.1, '1/km')
    print(att)