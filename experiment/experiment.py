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
        sol = self.__sol(z.m)
        return Power(sol[0], 'W')

    def _state_indices(self):
        pump_indices = []
        idx = 1
        for _ in self.pumps:
            pump_indices.append((idx, idx + 1))
            idx += 2
        return pump_indices

    def _raman_ode_system(self, z, P):
        Ps = P[0]
        dP = np.zeros_like(P)
        dPsdz = -self.fiber.alpha_s.m * Ps
        pump_indices = self._state_indices()

        for i, ((pf_idx, pb_idx), (pump_f, pump_b)) in enumerate(zip(pump_indices, self.pumps)):
            Ppf = P[pf_idx]
            Ppb = P[pb_idx]

            C_R_m = self.C_R(self.signal.wavelength, pump_f.wavelength) / 1e3
            wavelength_ratio_f = self.signal.wavelength.m / pump_f.wavelength.m
            wavelength_ratio_b = self.signal.wavelength.m / pump_b.wavelength.m

            dPsdz += C_R_m * Ps * (Ppf + Ppb)
            dP[pf_idx] = (
                -self.fiber.alpha_p.m * Ppf
                - wavelength_ratio_f * C_R_m * Ps * Ppf
            )
            dP[pb_idx] = (
                self.fiber.alpha_p.m * Ppb
                + wavelength_ratio_b * C_R_m * Ps * Ppb
            )
        dP[0] = dPsdz
        return dP

    def _boundary_conditions(self, ya, yb):
        res = np.zeros(1 + 2 * len(self.pumps))
        res[0] = ya[0] - self.signal.power.W
        pump_indices = self._state_indices()
        for i, ((pf_idx, pb_idx), (pump_f, pump_b)) in enumerate(zip(pump_indices, self.pumps)):
            res[pf_idx] = ya[pf_idx] - pump_f.power.W
            res[pb_idx] = yb[pb_idx] - pump_b.power.W
        return res

    def _initial_guess(self, z):
        y = np.zeros((1 + 2 * len(self.pumps), z.size))
        y[0] = self.signal.power.W
        pump_indices = self._state_indices()
        for (pf_idx, pb_idx), (pump_f, pump_b) in zip(pump_indices, self.pumps):
            y[pf_idx] = pump_f.power.W * np.exp(-self.fiber.alpha_p.m * z)
            y[pb_idx] = pump_b.power.W * np.exp(
                self.fiber.alpha_p.m * (z - self.fiber.length.m)
            )
        return y

    def _solve(self):
        z = np.linspace(0.0, self.fiber.length.m, 50)
        y_guess = self._initial_guess(z)

        sol = solve_bvp(
            self._raman_ode_system,
            self._boundary_conditions,
            x=z,
            y=y_guess,
        )

        if not sol.success:
            log.error("solve_bvp failed: %s", sol.message)

        return sol.sol


    def update(self):
        self.__sol = self._solve()
