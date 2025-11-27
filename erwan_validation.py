import copy
import numpy as np
import csv
import matplotlib.pyplot as plt

import custom_types as ct
import custom_types.conversions as conv
import raman_system as rs
import raman_amplifier as ra
import fibers as fib


def main():

    measured_on_off_gain_list: list[ct.PowerGain] = []
    wavelengths: list[ct.Length] = []
    with open("data/Raman_Gain_NF_stevan_Lezaja_November2025.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=';')

        for row in reader:
            if len(row) < 3:
                continue
            try:
                float(row[1])
            except ValueError:
                continue

            wavelength = float(row[1])
            on_off_gain = float(row[2])
            wavelengths.append(ct.Length(wavelength, 'nm'))
            measured_on_off_gain_list.append(ct.PowerGain(on_off_gain, 'dB'))

    input_spectrum = ra.Spectrum(ct.Power)
    for wl in wavelengths:
        freq = conv.wavelength_to_frequency(wl)
        input_spectrum.add_val(freq, ct.Power(250, 'uW'))


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
    print("Total input signal power:", ra.spectrum.integral(input_spectrum))
    print(raman_system.raman_inputs)
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
    measured_on_off_gain: ra.Spectrum[ct.PowerGain] = ra.Spectrum(ct.PowerGain, frequencies=simulated_gain_spectrum.frequencies, values=measured_on_off_gain_list)  # type: ignore

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))  # type: ignore

    ax.plot( # type: ignore
        [conv.frequency_to_wavelenth(f).nm for f in simulated_gain_spectrum.frequencies],
        [val.dB for val in simulated_gain_spectrum.values],
        label="Stevan's simulator"
    )
    ax.plot( # type: ignore
        [conv.frequency_to_wavelenth(f).nm for f in measured_on_off_gain.frequencies],
        [val.dB for val in measured_on_off_gain.values],
        label="Erwan's data"
    )
    ax.set_xlabel("Wavelength (nm)")  # type: ignore
    ax.set_ylabel("Power (dB)")  # type: ignore
    ax.set_ylim(bottom=-2, top=16) # type: ignore
    ax.set_title("ON/OFF Gain")  # type: ignore
    ax.grid()  # type: ignore
    ax.legend()  # type: ignore

    fig.show()

    _ = input()

if __name__ == "__main__":
    main()