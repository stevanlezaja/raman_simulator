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
from utils.loading_data_from_file import load_raman_dataset


def bar_plot_raman_inputs(ax: Axes, raman_inputs: ra.RamanInputs, predicted_inputs: ra.RamanInputs):  # type: ignore
    x = np.arange(6)
    width = 0.25
    ax.bar(x - width, raman_inputs.normalize().as_array(), width, label="Target")  # type: ignore
    ax.bar(x, predicted_inputs.normalize().as_array(), width, label="RPM")  # type: ignore


def plot_spectrums(ax: Axes, loop: cl.ControlLoop, predicted_spectrum: ra.Spectrum[ct.Power], real_spectrum: ra.Spectrum[ct.Power]):
    predicted_gain = predicted_spectrum/loop.off_power_spectrum
    real_gain = real_spectrum/loop.off_power_spectrum
    ax.plot( # type: ignore
        [f.Hz for f in predicted_gain.frequencies],
        [val.dB for val in predicted_gain.values],
        label="Predicted",
    )
    ax.plot( # type: ignore
        [f.Hz for f in real_gain.frequencies],
        [val.dB for val in real_gain.values],
        label="Simulated",
    )

    ymax = max(val.dB for val in predicted_gain.values)
    ymax = max(ymax, max(val.dB for val in real_gain.values))

    ax.set_xlabel("Frequency (Hz)")  # type: ignore
    ax.set_ylabel("Gain (dB)")  # type: ignore
    ax.set_ylim(0, 1.05 * ymax)
    ax.set_title("Target vs Current Output Spectrum")  # type: ignore
    ax.grid()  # type: ignore
    ax.legend()  # type: ignore



def main():
    raman_system = rs.RamanSystem()
    raman_system.fiber = fib.StandardSingleModeFiber(ct.Length(100, 'km'))
    raman_system.raman_amplifier = ra.RamanAmplifier(3, [0.0, 0.0, 0.0])

    input_spectrum = ra.Spectrum(ct.Power)
    for num in list(np.linspace(const.C_BAND[0], const.C_BAND[1], 40)):
        freq = conv.wavelength_to_frequency(ct.Length(num, 'nm'))
        input_spectrum.add_val(freq, ct.Power(25, 'uW'))
    raman_system.input_spectrum = input_spectrum
    raman_system.output_spectrum = copy.deepcopy(input_spectrum)

    controller = ctrl.PidController(0, 0, 0)
    loop = cl.ControlLoop(raman_system, controller)

    for raman_inputs, spectrum in load_raman_dataset('data/raman_simulator/3_pumps/100_fiber_0.0_ratio_sorted.json'):
        predicted_inputs = loop.inverse_model.get_raman_inputs(spectrum)
        print("Predicted", predicted_inputs)
        print("Real", raman_inputs)

        loop.curr_control = predicted_inputs
        loop.apply_control()
        predicted_spectrum = copy.deepcopy(loop.get_raman_output())

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))  # type: ignore

        bar_plot_raman_inputs(ax[0], raman_inputs, predicted_inputs)
        plot_spectrums(ax[1], loop, predicted_spectrum, spectrum)
        plt.show()  # type: ignore
        

if __name__ == '__main__':
    main()