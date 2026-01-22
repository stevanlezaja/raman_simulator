import copy

import numpy as np
import matplotlib.pyplot as plt

import raman_amplifier as ra
import raman_system as rs
import fibers as fib
import custom_types as ct
import custom_types.constants as const
import custom_types.conversions as conv


raman_system = rs.RamanSystem()
raman_system.fiber = fib.StandardSingleModeFiber(ct.Length(100, 'km'))

input_spectrum = ra.Spectrum(ct.Power)
for wl in np.linspace(const.C_BAND[0], const.C_BAND[1], 40):
    input_spectrum.add_val(
        conv.wavelength_to_frequency(ct.Length(wl, 'nm')),
        ct.Power(25, 'uW')
    )

raman_system.raman_amplifier = ra.RamanAmplifier(num_pumps=0, pumping_ratios=[])
raman_system.input_spectrum = copy.deepcopy(input_spectrum)
raman_system.output_spectrum = copy.deepcopy(input_spectrum)

raman_system.update()
off_spectrum = copy.deepcopy(raman_system.output_spectrum)

raman_system.raman_amplifier = ra.RamanAmplifier(num_pumps=1, pumping_ratios=[0])
raman_system.raman_amplifier.pump_powers = [ct.Power(200, 'mW')]
raman_system.raman_amplifier.pump_wavelengths = [ct.Length(1440, 'nm')]

raman_system.update()
single_pump_spectrum = copy.deepcopy(raman_system.output_spectrum)

raman_system.raman_amplifier = ra.RamanAmplifier(num_pumps=2, pumping_ratios=[0, 0])
raman_system.raman_amplifier.pump_powers = [ct.Power(200, 'mW'), ct.Power(200, 'mW')]
raman_system.raman_amplifier.pump_wavelengths = [ct.Length(1440, 'nm'), ct.Length(1470, 'nm')]

raman_system.update()
dual_pump_spectrum = copy.deepcopy(raman_system.output_spectrum)

raman_system.raman_amplifier = ra.RamanAmplifier(num_pumps=3, pumping_ratios=[0, 0, 0])
raman_system.raman_amplifier.pump_powers = [ct.Power(200, 'mW'), ct.Power(200, 'mW'), ct.Power(200, 'mW')]
raman_system.raman_amplifier.pump_wavelengths = [ct.Length(1430, 'nm'), ct.Length(1440, 'nm'), ct.Length(1470, 'nm')]

raman_system.update()
triple_pump_spectrum = copy.deepcopy(raman_system.output_spectrum)


single_gain = single_pump_spectrum / off_spectrum
dual_gain = dual_pump_spectrum / off_spectrum
triple_gain = triple_pump_spectrum / off_spectrum

plt.figure()
plt.plot(
    [f.THz for f in raman_system.output_spectrum.frequencies],
    [v.dB for v in single_gain.values],
)
plt.plot(
    [f.THz for f in raman_system.output_spectrum.frequencies],
    [v.dB for v in dual_gain.values],
)
plt.plot(
    [f.THz for f in raman_system.output_spectrum.frequencies],
    [v.dB for v in triple_gain.values],
)
plt.grid()
plt.savefig('plot.png')
