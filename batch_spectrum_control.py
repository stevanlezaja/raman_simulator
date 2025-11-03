from typing import Any
import itertools

import spectrum_control
import controllers as ctrl
import raman_system as rs
import custom_types as ct
import fibers as fib
import raman_amplifier as ra


def main():

    learning_rates: list[float] = [1e-1]
    weight_decays: list[float] = [0, 1e-5]
    betas: list[float] = [100, 500]
    gammas: list[float] = [0.999, 0.99]
    scale = [0.5, 1.0, 1.5]

    for params in itertools.product(learning_rates, weight_decays, betas, gammas, scale):
        params_dict: dict[str, Any]= {
            'lr': params[0],
            'weight_decay': params[1],
            'beta': params[2],
            'gamma': params[3],
            'power_step': ct.Power(10 * params[4], 'mW'),
            'wavelength_step': ct.Length(1 * params[4], 'nm'),
        }
        controller = ctrl.BernoulliController()
        controller.populate_parameters(params_dict)
        raman_system = rs.RamanSystem()
        fiber = fib.StandardSingleModeFiber()
        raman_system.fiber = fiber
        raman_system.raman_amplifier = ra.RamanAmplifier()

        spectrum_control.main(save_plots=True, iterations=500, raman_system=raman_system, controller=controller)


if __name__ == "__main__":
    main()