from .units import Unit


class Area(Unit):
    default_unit = 'm^2'

    def __init__(self, value: float, unit: str):
        super().__init__(value=value, unit=unit)

    @property
    def m(self):
        return self.value
    
    @m.setter
    def m(self, new: float):
        self.value = (new, 'm^2')

    @property
    def um(self):
        return self.value / 1e-12

    @um.setter
    def um(self, new: float):
        self.value = (new, 'um^2')


if __name__ == '__main__':
    a = Area(1, 'mm^2')
    print(a)

    a.m = 10
    print(a)

    a.um = 15
    print(a)