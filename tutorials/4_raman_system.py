import copy

import numpy as np
import matplotlib.pyplot as plt

import custom_types as ct
import custom_types.constants as const
import custom_types.conversions as conv
import raman_amplifier as ra
import raman_system as rs
import fibers as fib


# When creating a Raman system, you don't need to specify any parameters.
raman_system = rs.RamanSystem()

# You need to specify the fiber and Raman amplifier before using the system.
raman_system.fiber = fib.StandardSingleModeFiber(ct.Length(100, "km"))
raman_system.raman_amplifier = ra.RamanAmplifier(num_pumps=2, pumping_ratios=[0, 0])

# Now we need to make a spectrum that will contain the frequencies of interest.
# This spectrum will be an example 40 channel WDM spectrum in the C-band.
wdm_components_spectrum = ra.Spectrum(ct.Power)
# To do that, let's loop over 40 equidistant wavelengths in the C-band
for wl in np.linspace(const.C_BAND[0], const.C_BAND[1], 40):
    # And add a 100 uW component at each one
    wdm_components_spectrum.add_val(conv.wavelength_to_frequency(ct.Length(wl, 'nm')), ct.Power(100, 'uW'))

# Finally, we need to pass this spectrum as an input to our Raman system.
# Here we must make a deep copy of the object, since we don't want to modify the original object.
raman_system.input_spectrum = copy.deepcopy(wdm_components_spectrum)

# The output needs to be initialized to be a spectrum instance with all of the same frequencies as the input spectrum.
# This step is necessary due to poor architecture of the Raman system code.
raman_system.output_spectrum = copy.deepcopy(wdm_components_spectrum)

# Using the .update() method on the RamanSystem instance calculates the ODEs and updates the output spectrum.
raman_system.update()

# Now let's inspect the output spectrum by printing it.
print(raman_system.output_spectrum)

# And plotting it.
plt.figure()
plt.plot(
    [conv.frequency_to_wavelenth(f).nm for f in raman_system.output_spectrum.frequencies],
    [v.mW for v in raman_system.output_spectrum.values],
)
plt.savefig('plot1.png')
