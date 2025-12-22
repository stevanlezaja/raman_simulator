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
import raman_amplifier as ra
import controllers as ctrl
import models as m
from entry_points.train_models import get_or_train_backward_ensemble


log = clog.get_logger("Spectrum Control Test Script")

LOWER = 1500
UPPER = 1600
SAMPLES = 40

NUM_STEPS = 200

def _make_flat_spectrum(input_spectrum: ra.Spectrum[ct.Power]) -> ra.Spectrum[ct.Power]:
    target_spectrum = ra.Spectrum(ct.Power)
    for freq in input_spectrum.frequencies:
        target_spectrum.add_val(freq, ct.Power(50, 'uW'))
    return target_spectrum


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
        save_dir = f'plots/experiments/{controller.__class__.__name__}'
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(controller, ctrl.BernoulliController):
            exp_name = f"{iterations}steps_lr{controller.learning_rate}_wd{controller.weight_decay}_b{controller.beta}_g{controller.gamma}_ps{controller.power_step.mW}_ws{controller.wavelength_step.nm}_{number}.png"
        else:
            exp_name = 'experiment.png'
        exp_path = os.path.join(save_dir, exp_name)

    input_spectrum = ra.Spectrum(ct.Power)
    for num in list(np.linspace(wavelength_range[0], wavelength_range[1], num_samples)):
        freq = conv.wavelength_to_frequency(ct.Length(num, 'nm'))
        input_spectrum.add_val(freq, ct.Power(25, 'uW'))

    target_spectrum = _make_flat_spectrum(input_spectrum)

    raman_system.input_spectrum = input_spectrum
    raman_system.output_spectrum = copy.deepcopy(input_spectrum)

    control_loop = loop.ControlLoop(raman_system, controller)
    control_loop.set_target(target_spectrum)

    initial_powers: list[ct.Power] = []
    initial_wavelengths: list[ct.Length] = []
    for i in range(len(raman_system.raman_amplifier.pump_pairs)):
        initial_powers.append(ct.Power(np.random.uniform(low=ra.RamanInputs.MIN_POWER_W, high=ra.RamanInputs.MAX_POWER_W), 'W'))
        wl_low = ra.RamanInputs.MIN_WAVELENGTH_NM + i * (ra.RamanInputs.MAX_WAVELENGTH_NM - ra.RamanInputs.MIN_WAVELENGTH_NM) / len(raman_system.raman_amplifier.pump_pairs)
        wl_high = ra.RamanInputs.MIN_WAVELENGTH_NM + (i + 1) * (ra.RamanInputs.MAX_WAVELENGTH_NM - ra.RamanInputs.MIN_WAVELENGTH_NM) / len(raman_system.raman_amplifier.pump_pairs)
        initial_wavelengths.append(ct.Length(np.random.uniform(low=wl_low, high=wl_high), 'nm'))

    backward_model = m.BackwardEnsemble(get_or_train_backward_ensemble())
    import torch
    initial_input = ra.RamanInputs.from_array(backward_model.forward(torch.Tensor(target_spectrum.as_array())).detach().numpy())
    print(initial_input)

    control_loop.curr_control = initial_input

    if live_plot:
        plt.ion()  # type: ignore

    if live_plot or save_plots:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # type: ignore
        ax_err, ax_spec, ax_pow, ax_custom, ax_wl, ax_2d = axes.flatten()

    errors: list[float] = []
    powers: list[float] = []
    wavelengths: list[float] = []

    for curr_step in tqdm.tqdm(range(iterations)):
        final_step = curr_step == iterations - 1
        # if isinstance(control_loop.controller, ctrl.BernoulliController):
        #     if control_loop.controller.converged((0.48, 0.52), 50, 60):
        #         print("Converged")
        #         final_step = True

        control_loop.step()

        assert control_loop.curr_output is not None and control_loop.target is not None

        error = control_loop.curr_output - control_loop.target
        errors.append(abs(error.mean))

        # Log powers and wavelengths for plotting
        powers.append([p.W for p in control_loop.curr_control.powers])  # type: ignore
        wavelengths.append([w.nm for w in control_loop.curr_control.wavelengths])  # type: ignore

        # clear all axes
        if live_plot:
            for ax in axes.flatten():  # type: ignore
                ax.clear()

        if live_plot or (save_plots and final_step):
            control_loop.plot_loss(ax_err) # type: ignore
            control_loop.plot_spectrums(ax_spec)  # type: ignore
            control_loop.plot_parameter_2d(ax_2d)  # type: ignore
            control_loop.plot_power_evolution(ax_pow)  # type: ignore
            control_loop.plot_wavelength_evolution(ax_wl)  # type: ignore
            controller.plot_custom_data(ax_custom)  # type: ignore

            # update figure
            fig.tight_layout()  # type: ignore

        if live_plot:
            fig.canvas.draw()  # type: ignore
            fig.canvas.flush_events()  # type: ignore

        if final_step:
            break

    if live_plot:
        plt.ioff()  # type: ignore

    if save_plots:
        fig.savefig(exp_path, dpi=300)  # type: ignore
        log.info(f"Saved figure to {exp_path}")  # type: ignore
        plt.close(fig)  # type: ignore

    if live_plot or save_plots:
        plt.show()  # type: ignore


if __name__ == "__main__":
    main()
