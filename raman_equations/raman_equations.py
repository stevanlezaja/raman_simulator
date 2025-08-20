import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import solve_ivp


alpha_s = 1
alpha_p = 1

g_R = 1
a_p = 1

w_p = 1
w_s = 1


def raman_equations(z, P):
    Ps, Pp = P
    dPsdz = -alpha_s * Ps + g_R/a_p * Ps * Pp
    dPpdz = -alpha_p * Pp - w_p/w_s * g_R/a_p * Ps * Pp
    return [dPsdz, dPpdz]


z_min = 0
z_max = 100

Ps_initial = 10
Pp_initial = 10
initial_condition = [Ps_initial, Pp_initial]

z_points = np.linspace(z_min, z_max, 500)
solution = solve_ivp(raman_equations, [z_min, z_max], initial_condition, t_eval=z_points)

plt.figure()
plt.plot(solution.t, solution.y[0], label="Ps(z)")
plt.plot(solution.t, solution.y[1], label="Pp(z)")
plt.xlabel("z")
plt.ylabel("Power")
plt.legend()
plt.savefig("raman_plot.png", dpi=300)
plt.show()