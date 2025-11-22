import copy
import numpy as np
import scipy.io  # type: ignore
import matplotlib.pyplot as plt

import custom_types as ct
import custom_types.conversions as conv
import raman_system as rs
import raman_amplifier as ra
import fibers as fib


def main():
    input_spectrum = ra.Spectrum(ct.Power)
    for num in list(np.linspace(1530, 1560, 40)):
        freq = conv.wavelenth_to_frequency(ct.Length(num, 'nm'))
        input_spectrum.add_val(freq, ct.Power(25, 'uW'))


    num_pumps = 3
    pumping_ratios = [1.0 for _ in range(num_pumps)]
    raman_amplifier = ra.RamanAmplifier(num_pumps, pumping_ratios)

    raman_system = rs.RamanSystem()
    raman_system.raman_amplifier = raman_amplifier
    raman_system.fiber = fib.StandardSingleModeFiber(length=ct.Length(100, 'km'))
    raman_system.input_spectrum = input_spectrum
    raman_system.output_spectrum = copy.deepcopy(input_spectrum)
    raman_system.update()


    # Creating the ON spectrum
    raman_system.raman_inputs = ra.RamanInputs(
        powers=[ct.Power(200, 'mW'), ct.Power(200, 'mW'), ct.Power(200, 'mW')],
        wavelengths=[ct.Length(1455, 'nm'), ct.Length(1440, 'nm'), ct.Length(1420, 'nm')]
    )
    raman_system.update()

    simulated_power_spectrum_on = copy.deepcopy(raman_system.output_spectrum)

    # Creating the OFF spectrum
    raman_system.raman_inputs = ra.RamanInputs(
        powers=[ct.Power(0.0, 'W'), ct.Power(0.0, 'W'), ct.Power(0.0, 'W')],
        wavelengths=raman_system.raman_inputs.wavelengths
    )
    raman_system.update()

    simulated_power_spectrum_off = raman_system.output_spectrum

    # Calculating gain

    simulated_gain_spectrum: ra.Spectrum[ct.PowerGain] = simulated_power_spectrum_on / simulated_power_spectrum_off

    plt.ion()  # type: ignore

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))  # type: ignore

    ax.plot( # type: ignore
        [conv.frequency_to_wavelenth(f).nm for f in simulated_gain_spectrum.frequencies],
        [val.dB for val in simulated_gain_spectrum.values],
    )
    ax.set_xlabel("Frequency (Hz)")  # type: ignore
    ax.set_ylabel("Power (dB)")  # type: ignore
    ax.set_ylim(bottom=-2, top=16) # type: ignore
    ax.set_title("ON/OFF Gain")  # type: ignore
    ax.grid()  # type: ignore
    ax.legend()  # type: ignore

    fig.canvas.draw()  # type: ignore
    fig.canvas.flush_events()  # type: ignore

    _ = input()

if __name__ == "__main__":
    main()