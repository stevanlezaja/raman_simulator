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

def main(
        save_plots: bool = True,
        live_plot: bool = False,
        iterations: int = NUM_STEPS,
        num_samples: int = SAMPLES,
        wavelength_range: tuple[float, float] = (LOWER, UPPER),
        raman_system: rs.RamanSystem = rs.RamanSystem(),
        controller: ctrl.Controller = ctrl.BernoulliController(),
) -> None:

    if save_plots:
        save_dir = 'plots/experiments'
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(controller, ctrl.BernoulliController):
            exp_name = f"lr{controller.learning_rate}_wd{controller.weight_decay}_b{controller.beta}_g{controller.gamma}_ps{controller.power_step.mW}_ws{controller.wavelength_step.nm}.png"
        else:
            exp_name = 'experiment.png'
        exp_path = os.path.join(save_dir, exp_name)

    input_spectrum = ra.Spectrum(ct.Power)
    for num in list(np.linspace(wavelength_range[0], wavelength_range[1], num_samples)):
        freq = conv.wavelenth_to_frequency(ct.Length(num, 'nm'))
        input_spectrum.add_val(freq, ct.Power(10, 'mW'))

    target_spectrum = ra.Spectrum(ct.Power)
    target_power = ct.Power(0.5, 'W')
    target_power.mW = np.random.randint(low=250, high=750)
    target_wavelength = ct.Length(1450, 'nm')
    target_wavelength.nm = np.random.randint(low=1430, high=1470)
    # create values
    dummy_sytem = rs.RamanSystem()
    dummy_sytem.raman_amplifier = ra.RamanAmplifier()
    dummy_sytem.fiber = copy.deepcopy(raman_system.fiber)
    dummy_sytem.input_spectrum = copy.deepcopy(input_spectrum)
    dummy_sytem.output_spectrum = copy.deepcopy(input_spectrum)
    dummy_sytem.raman_amplifier.pump_power.mW = target_power.mW
    dummy_sytem.raman_amplifier.pump_wavelength.nm = target_wavelength.nm

    dummy_sytem.update()

    target_spectrum = dummy_sytem.output_spectrum

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
        ax_err, ax_spec, ax_pow, ax_p, ax_wl, ax_err_2d = axes.flatten()

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
            # --- Reward over time ---
            control_loop.plot_loss(ax_err) # type: ignore

            # --- Target vs Output spectrum ---
            ax_spec.plot(
                [f.Hz for f in control_loop.target.frequencies],
                [val.value for val in control_loop.target.values],
                label="Target",
            )  # type: ignore
            ax_spec.plot(
                [f.Hz for f in control_loop.curr_output.frequencies],
                [val.value for val in control_loop.curr_output.values],
                label="Current Output",
            )  # type: ignore
            ax_spec.set_xlabel("Frequency (Hz)")  # type: ignore
            ax_spec.set_ylabel("Power (mW)")  # type: ignore
            ax_spec.set_title("Target vs Current Output Spectrum")  # type: ignore
            ax_spec.grid()
            ax_spec.legend()  # type: ignore

            power_arr = np.array(powers)
            wl_arr = np.array(wavelengths)
            # --- Parameter evolution in 2D ---
            ax_err_2d.plot(power_arr, wl_arr)  # type: ignore
            ax_err_2d.scatter(power_arr[-1], wl_arr[-1], label="Current")  # type: ignore
            ax_err_2d.scatter(power_arr[0], wl_arr[0], label="Initial")  # type: ignore
            ax_err_2d.scatter(target_power.W, target_wavelength.nm, label="Target")  # type: ignore
            ax_err_2d.set_xlabel("Power [W]")  # type: ignore
            ax_err_2d.set_ylabel("Wavelength [nm]")  # type: ignore
            ax_err_2d.set_ylim([1420, 1490])
            ax_err_2d.set_xlim([0, 1])
            ax_err_2d.set_title("Wavelength step probability evolution")  # type: ignore
            ax_err_2d.grid()  # type: ignore
            ax_err_2d.legend()  # type: ignore

            # --- Power evolution ---
            for i in range(power_arr.shape[1]):
                ax_pow.plot(power_arr[::-1, i], range(len(power_arr[:, i])), label=f"Power {i}")  # type: ignore
            ax_pow.set_xlabel("Iteration")  # type: ignore
            ax_pow.set_ylabel("Power (W)")  # type: ignore
            ax_pow.set_xlim([0, 1])
            ax_pow.set_title("Power evolution")  # type: ignore
            ax_pow.grid()
            ax_pow.legend()  # type: ignore

            # --- Wavelength evolution ---
            for i in range(wl_arr.shape[1]):
                ax_wl.plot(wl_arr[:, i], label=f"Wavelength {i}")  # type: ignore
            ax_wl.set_xlabel("Iteration")  # type: ignore
            ax_wl.set_ylabel("Wavelength (nm)")  # type: ignore
            ax_wl.set_ylim([1420, 1490])
            ax_wl.set_title("Wavelength evolution")  # type: ignore
            ax_wl.grid()
            ax_wl.legend()  # type: ignore

            probs = np.array(control_loop.controller.history['probs'])  # shape: (steps, n_actions)
            # --- Step probability evolution ---
            ax_p.plot(probs[:, 0], label=f'Power')  # type: ignore
            ax_p.plot(probs[:, 1], label=f'Wavelength')  # type: ignore
            ax_p.set_xlabel("Iteration")  # type: ignore
            ax_p.set_ylabel("Probability")  # type: ignore
            ax_p.set_title("Step probability evolution")  # type: ignore
            ax_p.grid()
            ax_p.legend()  # type: ignore


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
