import functools
from scipy.integrate import solve_ivp, solve_bvp
import numpy as np

import custom_types.conversions as conv
from custom_types import Length, Frequency, Power
import custom_logging as clog

from fibers import Fiber
from signals import Signal
import raman_amplifier as ra


log = clog.get_logger("Experiment")


class Experiment:
    def __init__(self, fiber: Fiber, signal: Signal, pump_pair: tuple[ra.Pump, ra.Pump]):
        self.fiber = fiber
        self.signal = signal
        self.pump_wavelength = pump_pair[0].wavelength
        self.forward_pump_power = pump_pair[0].power
        self.backward_pump_power = pump_pair[1].power
        self.__sol = self._solve()

    @functools.cached_property
    def C_R(self):
        f_s = conv.wavelength_to_frequency(self.signal.wavelength)
        f_p = conv.wavelength_to_frequency(self.pump_wavelength)

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

    def _raman_ode_system(self, z, P):
        Ps, Ppf, Ppb = P
        C_R_m = self.C_R / 1e3
        dPsdz = -self.fiber.alpha_s.m * Ps + C_R_m * Ps * (Ppf + Ppb)
        dPpfdz = -self.fiber.alpha_p.m * Ppf - self.signal.wavelength.m / self.pump_wavelength.m * C_R_m * Ps * Ppf
        dPpbdz = self.fiber.alpha_p.m * Ppb + self.signal.wavelength.m / self.pump_wavelength.m * C_R_m * Ps * Ppb
        return np.vstack([dPsdz, dPpfdz, dPpbdz])

    def _solve_ivp(self):
        P0 = [self.signal.power.W, self.forward_pump_power.W, self.backward_pump_power.W]
        z_span = (0, self.fiber.length.m)
        num_points = 100
        z_eval = np.linspace(z_span[0], z_span[1], num_points)

        sol = solve_ivp(self._raman_ode_system, z_span, P0, t_eval=z_eval, dense_output=True)

        return sol.sol

    def _solve(self):
        def bc(ya, yb):
            res = np.zeros(3)
            res[0] = ya[0] - self.signal.power.W
            res[1] = ya[1] - self.forward_pump_power.W
            res[2] = yb[2] - self.backward_pump_power.W
            return res

        z_guess = np.linspace(0.0, self.fiber.length.m, 50)

        P_s_guess = np.full_like(z_guess, self.signal.power.W)
        P_p_plus_guess = self.forward_pump_power.W * np.exp(-self.fiber.alpha_p.m * z_guess)
        P_p_minus_guess = self.backward_pump_power.W * np.exp(self.fiber.alpha_p.m * (z_guess - self.fiber.length.m))
        y_guess = np.vstack([P_s_guess, P_p_plus_guess, P_p_minus_guess])

        sol = solve_bvp(self._raman_ode_system, bc, x = z_guess, y = y_guess)

        if not sol.success:
            log.error("solve_bvp failed: %s", sol.message)

        return sol.sol

    def update(self):
        self.__sol = self._solve()
