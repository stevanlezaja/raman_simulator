import matplotlib.pyplot as plt
import numpy as np
import copy

import raman_system as rs
import fibers as fib
import custom_types as ct
import custom_types.constants as const
import custom_types.conversions as conv
import raman_amplifier as ra
import controllers as ctrl
import signals as sig
import experiment as exp


raman_system = rs.RamanSystem()
raman_system.fiber = fib.StandardSingleModeFiber(ct.Length(100, 'km'))
raman_system.raman_amplifier = ra.RamanAmplifier(num_pumps=3)
raman_system.raman_amplifier.pump_wavelengths = [ct.Length(1450, 'nm'), ct.Length(1440, 'nm'), ct.Length(1475, 'nm')]
raman_system.raman_amplifier.pump_powers = [ct.Power(200, 'mW'), ct.Power(0, 'mW'), ct.Power(0, 'mW')]

controller = ctrl.PidController(0, 0, 0)

wavelength_range: tuple[float, float] = const.C_BAND

input_spectrum = ra.Spectrum(ct.Power)
freq = conv.wavelength_to_frequency(ct.Length(1550, 'nm'))
input_spectrum.add_val(freq, ct.Power(25, 'uW'))

raman_system.input_spectrum = input_spectrum
raman_system.output_spectrum = copy.deepcopy(input_spectrum)


plt.figure(figsize=(8, 6))  # type: ignore

power: list[ct.Power] = []
signal = sig.Signal()
signal.power = ct.Power(25, 'uW')
signal.wavelength = ct.Length(1550, 'nm')
experiment = exp.Experiment(raman_system.fiber, signal, raman_system.raman_amplifier.pump_pairs)
for dist in np.linspace(0, raman_system.fiber.length.m, 1000):
    power.append(experiment.get_signal_power_at_distance(ct.Length(dist, 'm')))

plt.plot([p.W for p in power])

plt.grid()  # type: ignore
plt.show()  # type: ignore