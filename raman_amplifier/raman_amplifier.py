from custom_types import Length, Power

class RamanAmplifier:
    def __init__(self):
        self.pump_wavelength = Length(1455, 'nm')
        self.pump_power = Power(0.5, 'W')
