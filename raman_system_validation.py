import os
import copy
import numpy as np
import scipy.io  # type: ignore
import matplotlib.pyplot as plt

import custom_types as ct
import custom_types.constants as c
import custom_types.conversions as conv
import raman_system as rs
import raman_amplifier as ra
import fibers as fib


def load_data(file_name: str, spectrum_frequencies: list[ct.Frequency]) -> tuple[list[ra.RamanInputs], list[ra.Spectrum[ct.PowerGain]]]:
    pumps = scipy.io.loadmat(file_name)  # type: ignore

    gain_cell = pumps['Gain_cell'].squeeze()  # type: ignore
    pump_pwer_cell = pumps['pump_pwer_cell'].squeeze() # type: ignore
    pump_wavelength_cell = pumps['pump_wavelength_cell'].squeeze() # type: ignore

    gain_list: list[list[float]] = [g.squeeze().tolist() for g in gain_cell] # type: ignore
    power_list: list[list[float]] = [p.squeeze().tolist() for p in pump_pwer_cell] # type: ignore
    wavelength_list: list[list[float]] = [w.squeeze().tolist() for w in pump_wavelength_cell] # type: ignore

    gain_list_ct: list[list[ct.PowerGain]] = [[ct.PowerGain(float(x), 'dB') for x in spectrum] for spectrum in gain_list.reverse()]
    power_list_ct: list[list[ct.Power]] = [[ct.Power(float(x), 'mW') for x in pumps] for pumps in power_list]
    wavelength_list_ct: list[list[ct.Length]] = [[ct.Length(float(x), 'nm') for x in pumps] for pumps in wavelength_list]

    assert len(power_list_ct) == len(wavelength_list)
    assert len(power_list_ct[0]) == len(wavelength_list[0])

    raman_inputs_list: list[ra.RamanInputs] = [ra.RamanInputs(powers, wavelengths) for powers, wavelengths in zip(power_list_ct, wavelength_list_ct)]

    spectrum_list: list[ra.Spectrum[ct.PowerGain]] = []
    for gains in gain_list_ct:
        assert len(gains) == len(spectrum_frequencies)

        spectrum = ra.Spectrum(ct.PowerGain)
        for freq, gain in zip(spectrum_frequencies, gains):
            spectrum.add_val(freq, gain)
        spectrum_list.append(spectrum)

    return (raman_inputs_list, spectrum_list)

file_name_1 = 'res_SMF_2pumps_Lspan100_Pch0dBm_5000runs_Dataset.mat'
file_name_2 = 'res_SMF_3pumps_Dataset.mat'
file_name_3 = 'res_SMF_3pumps_Lspan100_Pch0dBm_5000runs_Dataset.mat'

file_path = os.path.join('data', file_name_2)

input_spectrum = ra.Spectrum(ct.Power)
for num in list(np.linspace(c.C_BAND[0], c.C_BAND[1], 40)):
    freq = conv.wavelenth_to_frequency(ct.Length(num, 'nm'))
    input_spectrum.add_val(freq, ct.Power(100, 'mW'))

raman_inputs_list, spectrum_list = load_data(file_path, input_spectrum.frequencies)

num_pumps = 2 if file_name_1 in file_path else 3
pumping_ratios = [0.5 for _ in range(num_pumps)]
raman_amplifier = ra.RamanAmplifier(num_pumps, pumping_ratios)

raman_system = rs.RamanSystem()
raman_system.raman_amplifier = raman_amplifier
raman_system.fiber = fib.StandardSingleModeFiber(length=ct.Length(1, 'km'))
raman_system.input_spectrum = input_spectrum
raman_system.output_spectrum = copy.deepcopy(input_spectrum)
raman_system.update()

for raman_inputs, gain_spectrum in zip(raman_inputs_list, spectrum_list):
    raman_system.raman_inputs = raman_inputs
    raman_system.update()

    simulated_power_spectrum = raman_system.output_spectrum
    simulated_gain_spectrum = simulated_power_spectrum / input_spectrum

    error = ra.spectrum.mse(gain_spectrum, simulated_gain_spectrum)

    print(gain_spectrum.values[0])
    print(simulated_gain_spectrum.values[0].dB)






    plt.ion()  # type: ignore

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))  # type: ignore

    ax.plot( # type: ignore
        [f.Hz for f in gain_spectrum.frequencies],
        [val.dB for val in gain_spectrum.values],
        label="Measured",
    )
    ax.plot( # type: ignore
        [f.Hz for f in simulated_gain_spectrum.frequencies],
        [val.dB for val in simulated_gain_spectrum.values],
        label="Simulated",
    )
    ax.set_xlabel("Frequency (Hz)")  # type: ignore
    ax.set_ylabel("Power (dB)")  # type: ignore
    # ax.set_ylim( # type: ignore
    #     0,
    #     1.05 * max(
    #         max(val.dB for val in gain_spectrum.values),
    #         max(val.value for val in simulated_gain_spectrum.values),
    #     ),
    # )
    ax.set_title("Target vs Current Output Spectrum")  # type: ignore
    ax.grid()  # type: ignore
    ax.legend()  # type: ignore

    fig.canvas.draw()  # type: ignore
    fig.canvas.flush_events()  # type: ignore

    _ = input()
