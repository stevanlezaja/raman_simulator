import copy

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

def main() -> None:
    fiber = fib.StandardSingleModeFiber()
    fiber.length.km = 10

    raman_system = rs.RamanSystem()
    raman_system.raman_amplifier = ra.RamanAmplifier()
    raman_system.fiber = fiber

    input_spectrum = ra.Spectrum(ct.Power)
    for num in list(np.linspace(LOWER, UPPER, SAMPLES)):
        freq = conv.wavelenth_to_frequency(ct.Length(num, 'nm'))
        input_spectrum.add_val(freq, ct.Power(10, 'mW'))

    # parameters
    center_idx = 30         # peak position (sample index)
    peak_power = 200         # peak height in mW
    base_power = 10         # baseline in mW
    sigma = 8               # controls the smoothness (spread of peak)

    target_spectrum = ra.Spectrum(ct.Power)
    # create values
    for idx, num in enumerate(np.linspace(LOWER, UPPER, SAMPLES)):
        freq = conv.wavelenth_to_frequency(ct.Length(num, 'nm'))
        
        # Gaussian bump around center_idx
        bump = np.exp(-0.5 * ((idx - center_idx) / sigma) ** 2)
        power_mw = base_power + (peak_power - base_power) * bump
        
        target_spectrum.add_val(freq, ct.Power(power_mw, 'mW'))

    raman_system.input_spectrum = input_spectrum
    raman_system.output_spectrum = copy.deepcopy(input_spectrum)

    controller = ctrl.PidController(p=1, i=0, d=0)
    # controller = ctrl.BernoulliController()
    control_loop = loop.ControlLoop(raman_system, controller)
    control_loop.set_target(target_spectrum)
    control_loop.curr_control = ra.RamanInputs(powers=[ct.Power(0.5, 'W')], wavelengths=[ct.Length(1500, 'nm')])

    # enable interactive plotting
    plt.ion()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_err, ax_spec, ax_pow, ax_wl = axes.flatten()

    errors = []
    powers = []
    wavelengths = []

    for step in range(NUM_STEPS):
        print(f"\n\n STEP: {step}")
        control_loop.step()

        assert control_loop.curr_output is not None and control_loop.target is not None

        error = control_loop.curr_output - control_loop.target
        errors.append(abs(error.mean))

        # Log powers and wavelengths for plotting
        powers.append([p.W for p in control_loop.curr_control.powers])
        wavelengths.append([w.nm for w in control_loop.curr_control.wavelengths])

        # clear all axes
        for ax in axes.flatten():
            ax.clear()

        # --- subplot 1: Error over time ---
        ax_err.plot(errors, label="Error mean", color="tab:red")
        ax_err.set_xlabel("Iteration")
        ax_err.set_ylabel("Error (mean)")
        ax_err.set_title("Error over time")
        ax_err.legend()

        # --- subplot 2: Target vs Output spectrum ---
        ax_spec.plot(
            [f.Hz for f in control_loop.target.frequencies],
            [val.value for val in control_loop.target.values],
            label="Target",
        )
        ax_spec.plot(
            [f.Hz for f in control_loop.curr_output.frequencies],
            [val.value for val in control_loop.curr_output.values],
            label="Current Output",
        )
        ax_spec.set_xlabel("Frequency (Hz)")
        ax_spec.set_ylabel("Power (mW)")
        ax_spec.set_title("Target vs Current Output Spectrum")
        ax_spec.legend()

        # --- subplot 3: Power evolution ---
        power_arr = np.array(powers)
        for i in range(power_arr.shape[1]):
            ax_pow.plot(power_arr[:, i], label=f"Power {i}")
        ax_pow.set_xlabel("Iteration")
        ax_pow.set_ylabel("Power (W)")
        ax_pow.set_title("Power evolution")
        ax_pow.legend()

        # --- subplot 4: Wavelength evolution ---
        wl_arr = np.array(wavelengths)
        for i in range(wl_arr.shape[1]):
            ax_wl.plot(wl_arr[:, i], label=f"Wavelength {i}")
        ax_wl.set_xlabel("Iteration")
        ax_wl.set_ylabel("Wavelength (nm)")
        ax_wl.set_title("Wavelength evolution")
        ax_wl.legend()

        # update figure
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
