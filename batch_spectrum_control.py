import itertools

import spectrum_control
import controllers as ctrl
import raman_system as rs
import custom_types as ct
import fibers as fib
import raman_amplifier as ra


def main():

    learning_rates: list[float] = [1e-2]
    weight_decays: list[float] = [0, 1e-5]
    betas: list[float] = [100, 500, 1000, 5000, 10000]
    gammas: list[float] = [1, 0.99, 0.9, 0.75]
    power_steps = [ct.Power(1, 'mW')]
    wavelength_steps = [ct.Length(1, 'nm')]

    for params in itertools.product(learning_rates, weight_decays, betas, gammas, power_steps, wavelength_steps):
        params_dict = {
            'lr': params[0],
            'weight_decay': params[1],
            'beta': params[2],
            'gamma': params[3],
            'power_step': params[4],
            'wavelength_step': params[5],
        }
        controller = ctrl.BernoulliController()
        controller.populate_parameters(params_dict)
        raman_system = rs.RamanSystem()
        fiber = fib.StandardSingleModeFiber()
        raman_system.fiber = fiber
        raman_system.raman_amplifier = ra.RamanAmplifier()

        spectrum_control.main(save_plots=True, iterations=1000, raman_system=raman_system, controller=controller)


if __name__ == "__main__":
    main()