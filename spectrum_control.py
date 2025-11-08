import os
import copy
import tqdm

import numpy as np
import matplotlib.pyplot as plt

import custom_logging as clog
import custom_types as ct
import custom_types.conversions as conv

import control_loop as loop
import raman_system as rs
import fibers as fib
import raman_amplifier as ra
import controllers as ctrl


log = clog.get_logger("Spectrum Control Test Script")

LOWER = 1500
UPPER = 1600
SAMPLES = 40

NUM_STEPS = 200

def _make_flat_spectrum(input_spectrum: ra.Spectrum[ct.Power]) -> ra.Spectrum[ct.Power]:
    target_spectrum = ra.Spectrum(ct.Power)
    for freq in input_spectrum.frequencies:
        target_spectrum.add_val(freq, ct.Power(100, 'mW'))
    return target_spectrum

def _make_multipump_spectrum(
        raman_system: rs.RamanSystem,
        target_powers: list[ct.Power],
        target_wavelengths: list[ct.Length]
    ) -> ra.Spectrum[ct.Power]:

    dummy_sytem = rs.RamanSystem()
    dummy_sytem.raman_amplifier = ra.RamanAmplifier()
    dummy_sytem.fiber = copy.deepcopy(raman_system.fiber)
    dummy_sytem.input_spectrum = copy.deepcopy(raman_system.input_spectrum)
    dummy_sytem.output_spectrum = copy.deepcopy(raman_system.input_spectrum)
    dummy_sytem.raman_amplifier.pump_powers = target_powers
    dummy_sytem.raman_amplifier.pump_wavelengths = target_wavelengths
    dummy_sytem.update()

    return dummy_sytem.output_spectrum

def main(
        save_plots: bool = True,
        live_plot: bool = False,
        iterations: int = NUM_STEPS,
        num_samples: int = SAMPLES,
        wavelength_range: tuple[float, float] = (LOWER, UPPER),
        raman_system: rs.RamanSystem = rs.RamanSystem(),
        controller: ctrl.Controller = ctrl.BernoulliController(),
        number: int | None = None
) -> None:

    if save_plots:
        save_dir = 'plots/experiments'
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(controller, ctrl.BernoulliController):
            exp_name = f"lr{controller.learning_rate}_wd{controller.weight_decay}_b{controller.beta}_g{controller.gamma}_ps{controller.power_step.mW}_ws{controller.wavelength_step.nm}_{number}.png"
        else:
            exp_name = 'experiment.png'
        exp_path = os.path.join(save_dir, exp_name)

    input_spectrum = ra.Spectrum(ct.Power)
    for num in list(np.linspace(wavelength_range[0], wavelength_range[1], num_samples)):
        freq = conv.wavelenth_to_frequency(ct.Length(num, 'nm'))
        input_spectrum.add_val(freq, ct.Power(10, 'mW'))

    target_spectrum = _make_flat_spectrum(input_spectrum)

    raman_system.input_spectrum = input_spectrum
    raman_system.output_spectrum = copy.deepcopy(input_spectrum)

    control_loop = loop.ControlLoop(raman_system, controller)
    control_loop.set_target(target_spectrum)

    initial_power = ct.Power(0, 'W')
    initial_power.mW = np.random.randint(low=250, high=750)
    initial_wavelength = ct.Length(0, 'm')
    initial_wavelength.nm = np.random.randint(low=1430, high=1470)

    control_loop.curr_control = ra.RamanInputs(powers=[initial_power], wavelengths=[initial_wavelength])

    if live_plot:
        plt.ion()  # type: ignore

    if live_plot or save_plots:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # type: ignore
        ax_err, ax_spec, ax_pow, ax_custom, ax_wl, ax_2d = axes.flatten()

    errors: list[float] = []
    powers: list[float] = []
    wavelengths: list[float] = []

    for curr_step in tqdm.tqdm(range(iterations)):
        control_loop.step()

        assert control_loop.curr_output is not None and control_loop.target is not None

        error = control_loop.curr_output - control_loop.target
        errors.append(abs(error.mean))

        # Log powers and wavelengths for plotting
        powers.append([p.W for p in control_loop.curr_control.powers])
        wavelengths.append([w.nm for w in control_loop.curr_control.wavelengths])

        # clear all axes
        if live_plot:
            for ax in axes.flatten():
                ax.clear()

        if live_plot or (save_plots and curr_step == iterations - 1):
            control_loop.plot_loss(ax_err) # type: ignore
            control_loop.plot_spectrums(ax_spec)
            control_loop.plot_parameter_2d(ax_2d)
            control_loop.plot_power_evolution(ax_pow)
            control_loop.plot_wavelength_evolution(ax_wl)
            controller.plot_custom_data(ax_custom)

            # update figure
            fig.tight_layout()  # type: ignore

        if live_plot:
            fig.canvas.draw()  # type: ignore
            fig.canvas.flush_events()  # type: ignore

    if live_plot:
        plt.ioff()  # type: ignore

    if save_plots:
        fig.savefig(exp_path, dpi=300)
        log.info(f"Saved figure to {exp_path}")
        plt.close(fig)

    if live_plot or save_plots:
        plt.show()  # type: ignore


if __name__ == "__main__":
    main()
