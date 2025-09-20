from typing import Optional

from custom_types import Length, Power


class Pump:
    def __init__(self, power: Optional[Power] = None, wavelenght: Optional[Length] = None):
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
        self.forward_pump = Pump()
        self.backward_pump = Pump()
        self.update_pumps()

    def update_pumps(self):
        self.forward_pump.power = Power(self._pump_power.W * self._pumping_ratio, 'W')
        self.backward_pump.power = Power(self._pump_power.W * (1 - self._pumping_ratio), 'W')

    @property
    def pump_power(self):
        return self._pump_power

    @pump_power.setter
    def pump_power(self, new: Power):
        if new == self._pump_power:
            return
        self._pump_power = new
        self.update_pumps()

    @property
    def pumping_ratio(self):
        return self._pumping_ratio

    @pumping_ratio.setter
    def pumping_ratio(self, new: float):
        if new == self._pumping_ratio:
            return
        _validate_ratio(new)
        self._pumping_ratio = new
        self.update_pumps()

