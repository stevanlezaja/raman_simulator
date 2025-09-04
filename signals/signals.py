from custom_types import Length, Power


class Signal:
    def __init__(self):
        self.power = Power(0.1, 'mW')
        self.wavelength = Length(1470, 'nm')
