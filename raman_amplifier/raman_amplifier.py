from custom_types import Length, Power


class Pump:
    def __init__(self, power: Power, wavelenght: Length):
        self.wavelength = wavelenght
        self.power = power


class RamanAmplifier:
    def __init__(self, pumping_ratio: float=1.0):
        if pumping_ratio < 0 or pumping_ratio > 1.0:
            raise ValueError("Pumping ratio can only be in range [0.0, 1.0]")
        self.pump_power = Power(0.5, 'W')
        self.pump_wavelength = Length(1455, 'nm')
        self.forward_pump = Pump(Power(self.pump_power.W * pumping_ratio, 'W'), self.pump_wavelength)
        self.backward_pump = Pump(Power(self.pump_power.W * (1 - pumping_ratio), 'W'), self.pump_wavelength)
