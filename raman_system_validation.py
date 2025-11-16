import os
import numpy as np
import scipy.io

import custom_types as ct
import custom_types.constants as c
import custom_types.conversions as conv
import raman_system as rs
import raman_amplifier as ra


def load_data(file_name: str, spectrum_frequencies: list[ct.Frequency] = None) -> tuple[list[ra.RamanInputs], list[ra.Spectrum[ct.PowerGain]]]:
    pumps = scipy.io.loadmat(file_name)  # type: ignore

    gain_cell = pumps['Gain_cell'].squeeze()  # type: ignore
    pump_pwer_cell = pumps['pump_pwer_cell'].squeeze() # type: ignore
    pump_wavelength_cell = pumps['pump_wavelength_cell'].squeeze() # type: ignore

    gain_list: list[list[float]] = [g.squeeze().tolist() for g in gain_cell] # type: ignore

    gain_pg = [[ct.PowerGain(float(x), 'dB') for x in spectrum] for spectrum in gain_list]

    print(gain_pg)

    assert False

    reshaped_gain_cell = []
    reshaped_pump_pwer_cell = []
    reshaped_pump_wavelength_cell = []

    for i in range(len(gain_cell)):
        reshaped_gain_cell.append(gain_cell[i].squeeze())
        reshaped_pump_pwer_cell.append(pump_pwer_cell[i].squeeze())
        reshaped_pump_wavelength_cell.append(pump_wavelength_cell[i].squeeze())
        
    Y = torch.tensor(np.array(reshaped_gain_cell)).float()
    pump_pwer_cell = torch.tensor(np.array(reshaped_pump_pwer_cell))
    pump_wavelength_cell = torch.tensor(np.array(reshaped_pump_wavelength_cell))

    # Compute overall min and max
    overall_min_pwr = pump_pwer_cell.min().item()
    overall_max_pwr = pump_pwer_cell.max().item()
    overall_min_wavelen = pump_wavelength_cell.min().item()
    overall_max_wavelen = pump_wavelength_cell.max().item()

    # Set ranges: 3 powers, 3 wavelengths
    pump_pwr_ranges = [(overall_min_pwr, overall_max_pwr)] * pump_pwer_cell.shape[1]
    pump_wavelen_ranges = [(overall_min_wavelen, overall_max_wavelen)] * pump_wavelength_cell.shape[1]

    ranges = pump_pwr_ranges + pump_wavelen_ranges
    RaInputs.min_val = min(overall_min_pwr, overall_min_wavelen)
    RaInputs.max_val = max(overall_max_pwr, overall_max_wavelen)

    X = torch.cat((pump_pwer_cell, pump_wavelength_cell), dim=1).float()
        
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    X_train = [RaInputs(x) for x in X_train]
    X_test = [RaInputs(x) for x in X_test]
    Y_train = [GainSpectrum(x) for x in Y_train]
    Y_test = [GainSpectrum(x) for x in Y_test]
    
    return X_train, Y_train, X_test, Y_test, ranges


file_name_1 = 'res_SMF_2pumps_Lspan100_Pch0dBm_5000runs_Dataset.mat'
file_name_2 = 'res_SMF_3pumps_Dataset.mat'
file_name_3 = 'res_SMF_3pumps_Lspan100_Pch0dBm_5000runs_Dataset.mat'

file_path = os.path.join('data', file_name_2)

_ = load_data(file_path)