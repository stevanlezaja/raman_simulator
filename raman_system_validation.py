import os
import numpy as np
import scipy.io

import custom_types as ct
import custom_types.constants as c
import custom_types.conversions as conv
import raman_system as rs
import raman_amplifier as ra


def load_data(file_name: str, spectrum_frequencies: list[ct.Frequency]) -> tuple[list[ra.RamanInputs], list[ra.Spectrum[ct.PowerGain]]]:
    pumps = scipy.io.loadmat(file_name)  # type: ignore

    gain_cell = pumps['Gain_cell'].squeeze()  # type: ignore
    pump_pwer_cell = pumps['pump_pwer_cell'].squeeze() # type: ignore
    pump_wavelength_cell = pumps['pump_wavelength_cell'].squeeze() # type: ignore

    gain_list: list[list[float]] = [g.squeeze().tolist() for g in gain_cell] # type: ignore
    power_list: list[list[float]] = [p.squeeze().tolist() for p in pump_pwer_cell] # type: ignore
    wavelength_list: list[list[float]] = [w.squeeze().tolist() for w in pump_wavelength_cell] # type: ignore

    gain_list_ct: list[list[ct.PowerGain]] = [[ct.PowerGain(float(x), 'dB') for x in spectrum] for spectrum in gain_list]
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
    input_spectrum.add_val(freq, ct.Power(1000, 'mW'))

_ = load_data(file_path, input_spectrum.frequencies)
