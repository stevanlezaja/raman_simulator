from .units import Unit


class FiberAttenuation(Unit):
    default_unit = '1/m'

    def __init__(self, value: float, unit: str):
        super().__init__(value=value, unit=unit)

    @property
    def km(self):
        return self.m * 1e3

    @km.setter
    def km(self, new: float):
        self.m = new / 1e3

    @property
    def m(self):
        return self.value

    @m.setter
    def m(self, new: float):
        self.value = (new, '1/m')

    @property
    def dB_km(self):
        return 4.343 * self.km

    @dB_km.setter
    def dB_km(self, new: float):
        self.km = new / 4.343


if __name__ == "__main__":
    att = FiberAttenuation(0.1, '1/km')
    print(att)