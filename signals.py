from custom_types import Length


class Signal:
    def __init__(self):
        self.power = 1 * 1e-4
        self.wavelength = Length(1470, 'nm')
