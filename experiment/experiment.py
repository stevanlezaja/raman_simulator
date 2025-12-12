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
    def __init__(self, fiber: Fiber, signal: Signal, pump_pairs: list[tuple[ra.Pump, ra.Pump]]):
        self.fiber = fiber
        self.signal = signal
        self.pumps = pump_pairs
        self.__sol = self._solve()

    def C_R(self, signal_wavelength, pump_wavelength):
        f_s = conv.wavelength_to_frequency(signal_wavelength)
        f_p = conv.wavelength_to_frequency(pump_wavelength)

        freq_diff = Frequency(abs(f_p.Hz - f_s.Hz), 'Hz')

        return self.fiber.C_R(freq_diff)

    def get_signal_power_at_distance(self, z: Length) -> Power:
        assert isinstance(z, Length)
        Ps, _, _, _, _, _, _ = self.__sol(z.m)
        return Power(Ps, 'W')

    # def get_pump_power_at_distance(self, z: Length) -> Power:
    #     assert isinstance(z, Length)
    #     _, Ppf1, Ppb1 = self.__sol(z.m)
    #     return Power(Ppf1 + Ppb1, 'W')

    def _raman_ode_system(self, z, P):
        Ps, Ppf1, Ppb1, Ppf2, Ppb2, Ppf3, Ppb3 = P
        C_R_m1 = self.C_R(self.signal.wavelength, self.pumps[0][0].wavelength) / 1e3
        C_R_m2 = self.C_R(self.signal.wavelength, self.pumps[1][0].wavelength) / 1e3
        C_R_m3 = self.C_R(self.signal.wavelength, self.pumps[2][0].wavelength) / 1e3

        dPsdz = -self.fiber.alpha_s.m * Ps + C_R_m1 * Ps * (Ppf1 + Ppb1) + C_R_m2 * Ps * (Ppf2 + Ppb2) + C_R_m3 * Ps * (Ppf3 + Ppb3)

        dPpfdz1 = -self.fiber.alpha_p.m * Ppf1 - self.signal.wavelength.m / self.pumps[0][0].wavelength.m * C_R_m1 * Ps * Ppf1
        dPpbdz1 = self.fiber.alpha_p.m * Ppb1 + self.signal.wavelength.m / self.pumps[0][0].wavelength.m * C_R_m1 * Ps * Ppb1

        dPpfdz2 = -self.fiber.alpha_p.m * Ppf2 - self.signal.wavelength.m / self.pumps[0][0].wavelength.m * C_R_m2 * Ps * Ppf2
        dPpbdz2 = self.fiber.alpha_p.m * Ppb2 + self.signal.wavelength.m / self.pumps[0][0].wavelength.m * C_R_m2 * Ps * Ppb2

        dPpfdz3 = -self.fiber.alpha_p.m * Ppf3 - self.signal.wavelength.m / self.pumps[0][0].wavelength.m * C_R_m3 * Ps * Ppf3
        dPpbdz3 = self.fiber.alpha_p.m * Ppb3 + self.signal.wavelength.m / self.pumps[0][0].wavelength.m * C_R_m3 * Ps * Ppb3

        return np.vstack([dPsdz, dPpfdz1, dPpbdz1, dPpfdz2, dPpbdz2, dPpfdz3, dPpbdz3])

    def _solve(self):
        def bc(ya, yb):
            res = np.zeros(7)
            res[0] = ya[0] - self.signal.power.W
            res[1] = ya[1] - self.pumps[0][0].power.W
            res[2] = yb[2] - self.pumps[0][1].power.W
            res[3] = ya[3] - self.pumps[1][0].power.W
            res[4] = yb[4] - self.pumps[1][1].power.W
            res[5] = ya[5] - self.pumps[2][0].power.W
            res[6] = yb[6] - self.pumps[2][1].power.W
            return res

        z_guess = np.linspace(0.0, self.fiber.length.m, 50)

        P_s_guess = np.full_like(z_guess, self.signal.power.W)
        P_p1_plus_guess = self.pumps[0][0].power.W * np.exp(-self.fiber.alpha_p.m * z_guess)
        P_p1_minus_guess = self.pumps[0][1].power.W * np.exp(self.fiber.alpha_p.m * (z_guess - self.fiber.length.m))
        P_p2_plus_guess = self.pumps[1][0].power.W * np.exp(-self.fiber.alpha_p.m * z_guess)
        P_p2_minus_guess = self.pumps[1][1].power.W * np.exp(self.fiber.alpha_p.m * (z_guess - self.fiber.length.m))
        P_p3_plus_guess = self.pumps[2][0].power.W * np.exp(-self.fiber.alpha_p.m * z_guess)
        P_p3_minus_guess = self.pumps[2][1].power.W * np.exp(self.fiber.alpha_p.m * (z_guess - self.fiber.length.m))
        y_guess = np.vstack([P_s_guess, P_p1_plus_guess, P_p1_minus_guess, P_p2_plus_guess, P_p2_minus_guess, P_p3_plus_guess, P_p3_minus_guess])

        sol = solve_bvp(self._raman_ode_system, bc, x = z_guess, y = y_guess)

        if not sol.success:
            log.error("solve_bvp failed: %s", sol.message)

        return sol.sol

    def update(self):
        self.__sol = self._solve()
