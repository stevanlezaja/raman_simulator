from scipy.integrate import solve_ivp
import numpy as np
import functools

from custom_types import Length
from fibers import Fiber
from signals import Signal


class RamanAmplifier:
    def __init__(self, fiber: Fiber, signal: Signal):
        self.fiber = fiber
        self.signal = signal
        self.pump_wavelength = 1.455 * 1e-6  # [m]
        self.pump_power = 0.5  # [W]
        self.__sol = self.solve()

    @property
    def c(self):
        """Light speed"""
        return 3 * 1e8  # [m/s]

    @functools.cached_property
    def C_R(self):
        f_s = self.c / self.signal.wavelength  # [Hz]
        f_p = self.c / self.pump_wavelength  # [Hz]

        f_s = f_s / 1e12  # [THz]
        f_p = f_p / 1e12  # [THz]

        freq_diff = abs(f_p - f_s)

        return self.fiber.C_R(freq_diff)

    def get_signal_power_at_distance(self, z: Length):
        Ps, _ = self.__sol(z.m)
        return Ps

    def get_pump_power_at_distance(self, z: Length):
        _, Pp = self.__sol(z.m)
        return Pp

    def raman_ode_system(self, z, P):
        Ps, Pp = P
        dPsdz = -self.fiber.alpha_s * Ps + self.C_R * Ps * Pp
        dPpdz = -self.fiber.alpha_p * Pp - self.signal.wavelength / self.pump_wavelength * self.C_R * Ps * Pp
        return [dPsdz, dPpdz]
    
    def solve(self):
        P0 = [self.signal.power, self.pump_power]
        z_span = (0, self.fiber.length.m)
        num_points = 100
        z_eval = np.linspace(z_span[0], z_span[1], num_points)

        sol = solve_ivp(self.raman_ode_system, z_span, P0, t_eval=z_eval, dense_output=True)
        self.__sol = sol.sol

        return sol.sol