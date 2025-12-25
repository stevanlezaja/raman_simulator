from tqdm import tqdm
from typing import Any
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import custom_types as ct
import custom_types.constants as const
import custom_types.conversions as conv
import control_loop as cl
import raman_system as rs
import raman_amplifier as ra
import fibers as fib
import controllers as ctrl


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


# def plot_spectrums(ax: Axes, loop: cl.ControlLoop, target_spectrum: ra.Spectrum[ct.Power], real_spectrum: ra.Spectrum[ct.Power]):
#     target_gain = target_spectrum/loop.off_power_spectrum
#     real_gain = real_spectrum/loop.off_power_spectrum
#     ax.plot( # type: ignore
#         [f.Hz for f in target_gain.frequencies],
#         [val.dB for val in target_gain.values],
#         label="Target",
#     )
#     ax.plot( # type: ignore
#         [f.Hz for f in real_gain.frequencies],
#         [val.dB for val in real_gain.values],
#         label="Simulated",
#     )

#     ymax = max(val.dB for val in target_gain.values)
#     ymax = max(ymax, max(val.dB for val in real_gain.values))

#     ax.set_xlabel("Frequency (Hz)")  # type: ignore
#     ax.set_ylabel("Gain (dB)")  # type: ignore
#     ax.set_ylim(0, 1.05 * ymax)
#     ax.set_title("Target vs Current Output Spectrum")  # type: ignore
#     ax.grid()  # type: ignore
#     ax.legend()  # type: ignore


def plot_spectrums(
    ax: Axes,
    loop: cl.ControlLoop,
    target_spectrum: ra.Spectrum[ct.Power],
    real_spectrum: ra.Spectrum[ct.Power],
    *,
    color: Any,
    label: str | None = None,
):
    target_gain = target_spectrum / loop.off_power_spectrum
    real_gain = real_spectrum / loop.off_power_spectrum

    freqs = [f.Hz for f in target_gain.frequencies]

    ax.plot(  # type: ignore
        freqs,
        [val.dB for val in target_gain.values],
        color=color,
        linestyle="-",
        linewidth=2,
        label=label,
    )

    ax.plot(  # type: ignore
        freqs,
        [val.dB for val in real_gain.values],
        color=color,
        linestyle="--",
        linewidth=2,
    )


# def main():
#     raman_system = rs.RamanSystem()
#     raman_system.fiber = fib.StandardSingleModeFiber(ct.Length(100, 'km'))
#     raman_system.raman_amplifier = ra.RamanAmplifier(3, [0.0, 0.0, 0.0])

#     input_spectrum = ra.Spectrum(ct.Power)
#     for num in list(np.linspace(const.C_BAND[0], const.C_BAND[1], 40)):
#         freq = conv.wavelength_to_frequency(ct.Length(num, 'nm'))
#         input_spectrum.add_val(freq, ct.Power(25, 'uW'))
#     raman_system.input_spectrum = input_spectrum
#     raman_system.output_spectrum = copy.deepcopy(input_spectrum)

#     controller = ctrl.PidController(0, 0, 0)
#     loop = cl.ControlLoop(raman_system, controller)

#     for gain in range(5, 14):

#         target_spectrum = _make_flat_spectrum(loop.off_power_spectrum, ct.PowerGain(gain, 'dB'))

#         predicted_inputs = loop.inverse_model.get_raman_inputs(target_spectrum)

#         loop.curr_control = predicted_inputs
#         loop.apply_control()
#         predicted_spectrum = copy.deepcopy(loop.get_raman_output())

#         fig, ax = plt.subplots(1, 2, figsize=(8, 4))  # type: ignore

#         plot_spectrums(ax[1], loop, target_spectrum, predicted_spectrum)
#         plt.show()  # type: ignore
    

def main():
    raman_system = rs.RamanSystem()
    raman_system.fiber = fib.StandardSingleModeFiber(ct.Length(100, 'km'))
    raman_system.raman_amplifier = ra.RamanAmplifier(3, [0.0, 0.0, 0.0])

    input_spectrum = ra.Spectrum(ct.Power)
    for num in np.linspace(const.C_BAND[0], const.C_BAND[1], 40):
        freq = conv.wavelength_to_frequency(ct.Length(num, 'nm'))
        input_spectrum.add_val(freq, ct.Power(25, 'uW'))

    raman_system.input_spectrum = input_spectrum
    raman_system.output_spectrum = copy.deepcopy(input_spectrum)

    controller = ctrl.PidController(0, 0, 0)
    loop = cl.ControlLoop(raman_system, controller)

    _, ax = plt.subplots(figsize=(8, 4))  # type: ignore

    for gain in tqdm(range(5, 14)):
        target_spectrum = _make_flat_spectrum(
            loop.off_power_spectrum,
            ct.PowerGain(gain, 'dB')
        )

        predicted_inputs = loop.inverse_model.get_raman_inputs(target_spectrum)

        loop.curr_control = predicted_inputs
        loop.apply_control()
        predicted_spectrum = copy.deepcopy(loop.get_raman_output())

        line = ax.plot([], [])[0]  # type: ignore
        color = line.get_color()
        line.remove()

        plot_spectrums(
            ax,
            loop,
            target_spectrum,
            predicted_spectrum,
            color=color,
        )

    ax.set_xlabel("Frequency (Hz)")  # type: ignore
    ax.set_ylabel("Gain (dB)")  # type: ignore
    ax.set_title("Flat Target vs Achieved Gain (Inverse Model)")  # type: ignore
    ax.grid(True)  # type: ignore

    plt.show()  # type: ignore


if __name__ == '__main__':
    main()