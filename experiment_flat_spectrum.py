import os
from typing import Any
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

import custom_types as ct
import custom_types.constants as const
import custom_types.conversions as conv
import control_loop as cl
import raman_system as rs
import raman_amplifier as ra
import fibers as fib
import controllers as ctrl
from entry_points import spectrum_control


def _make_flat_spectrum(off_spectrum: ra.Spectrum[ct.Power], target_val: ct.Power | ct.PowerGain | Any) -> ra.Spectrum[ct.Power]:
    def _get_power(input_power: ct.Power, power_gain: ct.PowerGain) -> ct.Power:
        pout = input_power.value * power_gain.linear
        return ct.Power(pout, input_power.default_unit)

    if isinstance(target_val, ct.PowerGain):
        target_power = _get_power(off_spectrum.values[0], target_val)
    elif isinstance(target_val, ct.Power):
        target_power = target_val
    else:
        raise ValueError(f"Target can be either Power or PowerGain, and not {type(target_val)}")

    target_spectrum = ra.Spectrum(ct.Power)
    for freq in off_spectrum.frequencies:
        target_spectrum.add_val(freq, target_power)
    return target_spectrum


def plot_spectrums(
    ax: Axes,
    loop: cl.ControlLoop,
    target_spectrum: ra.Spectrum[ct.Power],
    initial_spectrum: ra.Spectrum[ct.Power],
    bern_fine_tuned_spectrum: ra.Spectrum[ct.Power],
    gd_fine_tuned_spectrum: ra.Spectrum[ct.Power],
    *,
    color: Any,
    label: str | None = None,
):
    target_gain = target_spectrum / loop.off_power_spectrum
    initial_gain = initial_spectrum / loop.off_power_spectrum
    bern_fine_tuned_gain = bern_fine_tuned_spectrum / loop.off_power_spectrum
    gd_fine_tuned_gain = gd_fine_tuned_spectrum / loop.off_power_spectrum

    freqs = [f.Hz for f in target_gain.frequencies]
    ax.set_ylim(bottom=0, top=15)

    ax.plot(  # type: ignore
        freqs,
        [val.dB for val in target_gain.values],
        color=color,
        linestyle=":",
        linewidth=2,
        label=label,
    )

    ax.plot(  # type: ignore
        freqs,
        [val.dB for val in initial_gain.values],
        color=color,
        linestyle="--",
        linewidth=2,
    )

    ax.plot(  # type: ignore
        freqs,
        [val.dB for val in bern_fine_tuned_gain.values],
        color=color,
        linestyle="-",
        linewidth=2,
    )

    ax.plot(  # type: ignore
        freqs,
        [val.dB for val in gd_fine_tuned_gain.values],
        color=color,
        linestyle="-.",
        linewidth=2,
    )


def main():
    bern_raman_system = rs.RamanSystem()
    bern_raman_system.fiber = fib.StandardSingleModeFiber(ct.Length(100, 'km'))
    bern_raman_system.raman_amplifier = ra.RamanAmplifier(3, [0.0, 0.0, 0.0])

    input_spectrum = ra.Spectrum(ct.Power)
    for num in np.linspace(const.C_BAND[0], const.C_BAND[1], 40):
        freq = conv.wavelength_to_frequency(ct.Length(num, 'nm'))
        input_spectrum.add_val(freq, ct.Power(25, 'uW'))

    bern_raman_system.input_spectrum = input_spectrum
    bern_raman_system.output_spectrum = copy.deepcopy(input_spectrum)

    bern_controller = ctrl.BernoulliController(
        lr=1e-1,
        power_step=ct.Power(1, 'mW'),
        wavelength_step=ct.Length(1, 'nm'),
        beta=10,
        gamma=0.99,
        weight_decay=1e-3,
        input_dim=6,
    )
    bern_loop = cl.ControlLoop(bern_raman_system, bern_controller)

    gd_raman_system = copy.deepcopy(bern_raman_system)
    gd_controller = ctrl.GradientDescentController(
        training_data='controllers/gradient_descent_controller/data/raman_simulator_3_pumps_0.0_ratio.json',
        epochs=100,
        lr_control=10
    )
    gd_loop = cl.ControlLoop(gd_raman_system, gd_controller)

    plt.ion()  # type: ignore
    fig, ax = plt.subplots(figsize=(8, 4))  # type: ignore

    ax.set_xlabel("Frequency (Hz)")  # type: ignore
    ax.set_ylabel("Gain (dB)")  # type: ignore
    ax.set_title("Flat Target vs Achieved Gain (Inverse Model)")  # type: ignore
    ax.grid(True)  # type: ignore

    legend_elements = [
        Line2D([0], [0], color="black", linestyle=":", linewidth=2, label="Target"),
        Line2D([0], [0], color="black", linestyle="-.", linewidth=2, label="GD"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Initial"),
        Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="Bernoulli"),
    ]

    ax.legend(handles=legend_elements, loc="best")  # type: ignore

    for target_gain in [5, 7, 9, 11, 13]:
        print(f"Optimizing for flat spectrum at {target_gain} dB")
        target_spectrum = _make_flat_spectrum(
            bern_loop.off_power_spectrum,
            ct.PowerGain(target_gain, 'dB')
        )

        predicted_inputs = bern_loop.inverse_model.get_raman_inputs(target_spectrum)

        bern_loop.curr_control = predicted_inputs
        bern_loop.apply_control()
        bern_loop.set_target(target_spectrum)

        gd_loop.curr_control = predicted_inputs
        gd_loop.apply_control()
        gd_loop.set_target(target_spectrum)

        initial_spectrum = copy.deepcopy(bern_loop.get_raman_output())

        spectrum_control.main(
            save_plots=False,
            iterations=50,
            control_loop=bern_loop,
        )
        bern_fine_tuned_spectrum = copy.deepcopy(bern_loop.curr_output)
        assert bern_fine_tuned_spectrum is not None

        spectrum_control.main(
            save_plots=False,
            iterations=50,
            control_loop=gd_loop,
        )
        gd_fine_tuned_spectrum = copy.deepcopy(gd_loop.curr_output)
        assert gd_fine_tuned_spectrum is not None

        line = ax.plot([], [])[0]  # type: ignore
        color = line.get_color()
        line.remove()

        plot_spectrums(
            ax,
            bern_loop,
            target_spectrum,
            initial_spectrum,
            bern_fine_tuned_spectrum,
            gd_fine_tuned_spectrum,
            color=color,
        )

        ax.relim()
        ax.autoscale_view()

        fig.canvas.draw()  # type: ignore
        fig.canvas.flush_events()
        plt.pause(0.01)

    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/flat_gain_inverse_model.png", dpi=300)  # type: ignore

    plt.show()  # type: ignore


if __name__ == '__main__':
    main()