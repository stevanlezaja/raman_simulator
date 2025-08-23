import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import solve_ivp


alpha_s = 4.605 * 1e-5  # [m^-1]
alpha_p = 4.605 * 1e-5  # [m^-1]

g_R = 6 * 1e-14  # [m/W]
a_p = 50 * 1e-12  # [m^2]

w_p = 1
w_s = 1

lambda_s = 1550 * 1e-9  # [m]
lambda_p = 1455 * 1e-9  # [m]

w_ratio = lambda_s / lambda_p


def raman_ode_system(z, P):
    Ps, Pp = P
    dPsdz = -alpha_s * Ps + g_R/a_p * Ps * Pp
    dPpdz = -alpha_p * Pp - w_ratio * g_R/a_p * Ps * Pp
    return [dPsdz, dPpdz]


z_min = 0
z_max = 25 * 1e3  # [m]

Ps_initial = 1 * 1e-4  # [W]
Pp_initial = 0.5  # [W]

initial_condition = [Ps_initial, Pp_initial]

z_points = np.linspace(z_min, z_max, 500)
solution = solve_ivp(raman_ode_system, [z_min, z_max], initial_condition, t_eval=z_points)

plt.figure()
plt.plot(solution.t, solution.y[0], label="Ps(z)")
plt.plot(solution.t, solution.y[1], label="Pp(z)")
plt.xlabel("z")
plt.ylabel("Power")
plt.legend()
plt.savefig("raman_plot.png", dpi=300)
plt.show()