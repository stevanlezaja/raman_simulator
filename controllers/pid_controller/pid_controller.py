"""
This module contains the implementation of PID Controller class
"""

import raman_amplifier as ra
import custom_types as ct

from ..controller_base import Controller

class PidController(Controller):
    """
    Implements PID Control strategy
    """
    def __init__(self, p: float=0.2, i: float=0.1, d: float=0.1):
        self._params['p'] = (float, p)
        self._params['i'] = (float, i)
        self._params['d'] = (float, d)
        self.integral = 0.0
        self.e1 = 0.0

    def get_control(
            self,
            curr_input: ra.RamanInputs,
            curr_output: ra.Spectrum[ct.Power],
            target_output: ra.Spectrum[ct.Power]
        ) -> ra.RamanInputs:

        power_control = self._get_power_control(target_output, curr_output)
        wavelength_control = self._get_wavelength_control(target_output, curr_output)

        delta_powers = [ct.Power(power_control, 'W') for _ in curr_input.powers]
        delta_wavelengths = [ct.Length(wavelength_control, 'nm') for _ in curr_input.wavelengths]

        return ra.RamanInputs(powers=delta_powers, wavelengths=delta_wavelengths)

    @property
    def p(self):
        return self._params['p'][1]

    @property
    def i(self):
        return self._params['i'][1]

    @property
    def d(self):
        return self._params['d'][1]

    def _get_power_control(self, target_output: ra.Spectrum[ct.Power], curr_output: ra.Spectrum[ct.Power]) -> float:
        error = (target_output - curr_output).mean
        return self._pid(error)

    def _get_wavelength_control(self, target_output: ra.Spectrum[ct.Power], curr_output: ra.Spectrum[ct.Power]) -> float:
        curr_peak = curr_output.peak_frequency()
        target_peak = target_output.peak_frequency()

        error = -(target_peak - curr_peak).THz

        return self._pid(error)

    def _pid(self, e: float) -> float:
        p = self.p * e

        self.integral += self.i * e
        i = self.integral

        d = self.d * (e - self.e1)
        self.e1 = e

        return p + i + d

    def update_controller(
            self,
            error: ra.Spectrum[ct.Power],
            control_delta: ra.RamanInputs
        ) -> None:
        pass
