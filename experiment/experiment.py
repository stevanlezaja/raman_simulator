import functools
from scipy.integrate import solve_ivp, solve_bvp
import numpy as np

import custom_types.conversions as conv
from custom_types import Length, Frequency, Power

from fibers import Fiber
from signals import Signal
from raman_amplifier import RamanAmplifier


class Experiment:
    def __init__(self, fiber: Fiber, signal: Signal, raman_amplifier: RamanAmplifier):
        self.fiber = fiber
        self.signal = signal
        self.raman_amplifier = raman_amplifier
        self.__sol = self._solve()

    @functools.cached_property
    def C_R(self):
        f_s = conv.wavelenth_to_frequency(self.signal.wavelength)
        f_p = conv.wavelenth_to_frequency(self.raman_amplifier.pump_wavelength)

        freq_diff = Frequency(abs(f_p.Hz - f_s.Hz), 'Hz')

        return self.fiber.C_R(freq_diff)

    def get_signal_power_at_distance(self, z: Length) -> Power:
        assert isinstance(z, Length)
        Ps, _, _ = self.__sol(z.m)
        return Power(Ps, 'W')

    def get_pump_power_at_distance(self, z: Length) -> Power:
        assert isinstance(z, Length)
        _, Ppf, Ppb = self.__sol(z.m)
        return Power(Ppf + Ppb, 'W')

    def raman_ode_system(self, z, P):
        Ps, Ppf, Ppb = P
        C_R_m = self.C_R / 1e3
        dPsdz = -self.fiber.alpha_s * Ps + C_R_m * Ps * (Ppf + Ppb)
        dPpfdz = -self.fiber.alpha_p * Ppf - self.signal.wavelength.m / self.raman_amplifier.pump_wavelength.m * C_R_m * Ps * Ppf
        dPpbdz = -(-self.fiber.alpha_p * Ppb - self.signal.wavelength.m / self.raman_amplifier.pump_wavelength.m * C_R_m * Ps * Ppb)
        return [dPsdz, dPpfdz, dPpbdz]
    
    def solve(self):
        P0 = [self.signal.power.W, self.raman_amplifier.forward_pump.power.W, self.raman_amplifier.backward_pump.power.W]
        z_span = (0, self.fiber.length.m)
        num_points = 100
        z_eval = np.linspace(z_span[0], z_span[1], num_points)

        sol = solve_ivp(self.raman_ode_system, z_span, P0, t_eval=z_eval, dense_output=True)
        self.__sol = sol.sol

        return sol.sol