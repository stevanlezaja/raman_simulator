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
        self.p = p
        self.i = i
        self.d = d
        self.integral = 0.0
        self.e1 = 0.0

    def get_control(
            self,
            curr_input: ra.RamanInputs,
            curr_output: ra.Spectrum[ct.Power],
            target_output: ra.Spectrum[ct.Power]
        ) -> ra.RamanInputs:

        e = (target_output - curr_output).mean

        p = self.p * e

        self.integral += self.i * e
        i = self.integral

        d = self.d * (e - self.e1)
        self.e1 = e

        control = p + i + d

        # delta_wavelengths = [ct.Length(control, 'nm') for _ in curr_input.powers]
        delta_powers = [ct.Power(control, 'W') for _ in curr_input.powers]
        delta_wavelengths = [ct.Length(0.0, 'm') for _ in curr_input.wavelengths]

        print(delta_wavelengths)
        print(delta_powers)

        return ra.RamanInputs(powers=delta_powers, wavelengths=delta_wavelengths)

    def update_controller(
            self,
            error: ra.Spectrum[ct.Power],
            control_delta: ra.RamanInputs
        ) -> None:
        pass
