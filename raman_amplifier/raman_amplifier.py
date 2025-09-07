from custom_types import Length, Power


class Pump:
    def __init__(self, power: Power, wavelenght: Length):
        self.wavelength = wavelenght
        self.power = power


def _validate_ratio(pumping_ratio: float):
    if pumping_ratio < 0 or pumping_ratio > 1.0:
        raise ValueError("Pumping ratio can only be in range [0.0, 1.0]")


class RamanAmplifier:
    def __init__(self, pumping_ratio: float=1.0):
        _validate_ratio(pumping_ratio)
        self._pumping_ratio = pumping_ratio
        self._pump_power = Power(0.5, 'W')
        self.pump_wavelength = Length(1455, 'nm')
        self.forward_pump = Pump(Power(self._pump_power.W * self.pumping_ratio, 'W'), self.pump_wavelength)
        self.backward_pump = Pump(Power(self._pump_power.W * (1 - self.pumping_ratio), 'W'), self.pump_wavelength)

    def update_pumps(self):
        self.forward_pump.power = Power(self._pump_power.W * self._pumping_ratio, 'W')
        self.backward_pump.power = Power(self._pump_power.W * (1 - self._pumping_ratio), 'W')
    def pump_power(self, new: Power):
        self._pump_power = new
        self.forward_pump.power = Power(self._pump_power.W * self.pumping_ratio, 'W')
        self.backward_pump.power = Power(self._pump_power.W * (1 - self.pumping_ratio), 'W')

