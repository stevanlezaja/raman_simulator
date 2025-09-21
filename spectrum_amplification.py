import numpy as np
import matplotlib.pyplot as plt

import custom_types as ct
import custom_types.conversions as conv

import raman_amplifier as ra
import fibers as fib
import signals as sig
import experiment as exp


wavelengths = list(np.linspace(1450, 1600, 40))

input_spectrum = ra.io.Spectrum(ct.Power)

for wl in wavelengths:
    power = ct.Power(value=1, unit='mW')
    input_spectrum.add_val(conv.wavelenth_to_frequency(ct.Length(wl, 'nm')), power)

print(input_spectrum)
amplifier = ra.RamanAmplifier(pumping_ratio=0.5)
amplifier.pump_power.W = 1.0
amplifier.pump_wavelength.nm = 1450

fiber = fib.StandardSingleModeFiber()
fiber.length.km = 10

output_components: list[sig.Signal] = []

for component in input_spectrum:
    s = sig.Signal()
    s.wavelength = conv.frequency_to_wavelenth(component[0])
    s.power = component[1]

    experiment = exp.RamanSystem(fiber, s, amplifier)

    output_signal = sig.Signal()
    output_signal.wavelength = conv.frequency_to_wavelenth(component[0])
    output_signal.power = experiment.get_signal_power_at_distance(fiber.length)

    output_components.append(output_signal)

output_spectrum = ra.io.Spectrum(ct.Power)

for comp in output_components:
    output_spectrum.add_val(conv.wavelenth_to_frequency(comp.wavelength), comp.power)


x_axis = [f.THz for f in list(output_spectrum.frequencies)]
y_out = [p.mW for p in list(output_spectrum.values)]
y_in = [p.mW for p in list(input_spectrum.values)]

fig, ax = plt.subplots() # type: ignore
ax.plot(x_axis, y_in) # type: ignore
ax.plot(x_axis, y_out) # type: ignore
plt.show() # type: ignore

