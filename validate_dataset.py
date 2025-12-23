from copy import deepcopy

import custom_types as ct
from utils.loading_data_from_file import load_raman_dataset

dataset = load_raman_dataset('data/raman_simulator/3_pumps/100_fiber_0.0_ratio_sorted.json')

min_power = ct.Power(1000, 'W')
max_power = ct.Power(-100, 'W')
min_wl = ct.Length(1e12, 'm')   # very large
max_wl = ct.Length(-1e12, 'm')  # very small


for ra_in, spec in dataset:
    # Check powers
    if any(p < min_power for p in ra_in.powers):
        min_power = deepcopy(min(ra_in.powers))
        print("Found new min power:", min_power.mW)
    if any(p > max_power for p in ra_in.powers):
        max_power = deepcopy(max(ra_in.powers))
        print("Found new max power:", max_power.mW)
    
    # Check wavelengths
    if any(w < min_wl for w in ra_in.wavelengths):
        min_wl = deepcopy(min(ra_in.wavelengths))
        print("Found new min wavelength:", min_wl.nm)
    if any(w > max_wl for w in ra_in.wavelengths):
        max_wl = deepcopy(max(ra_in.wavelengths))
        print("Found new max wavelength:", max_wl.nm)

print("Final range of the dataset:")
print(f"Power range: [{min_power}:{max_power}]")
print(f"Wavelength range: [{min_wl}:{max_wl}]")