import copy

import matplotlib.pyplot as plt
import numpy as np

import custom_types as ct
import custom_types.constants as const
import custom_types.conversions as conv
import raman_amplifier as ra
import raman_system as rs
import fibers as fib
import controllers as ctrl
import control_loop as cl


# In this tutorial we will focus on making a Control Loop.

# First, we need to make a Raman system as in the previous tutorial.
raman_system = rs.RamanSystem()
raman_system.fiber = fib.StandardSingleModeFiber(ct.Length(100, "km"))
raman_system.raman_amplifier = ra.RamanAmplifier(num_pumps=2, pumping_ratios=[0, 0])

wdm_components_spectrum = ra.Spectrum(ct.Power)
for wl in np.linspace(const.C_BAND[0], const.C_BAND[1], 40):
    wdm_components_spectrum.add_val(conv.wavelength_to_frequency(ct.Length(wl, 'nm')), ct.Power(100, 'uW'))

raman_system.input_spectrum = copy.deepcopy(wdm_components_spectrum)
raman_system.output_spectrum = copy.deepcopy(wdm_components_spectrum)

# Then we need to make a Controller
# There are many different types of controllers, and their base class is designed for easy implementation of new controllers.
# For now we will use the Bernoulli controller and focus on the controller details later.
controller = ctrl.BernoulliController(
    input_dim=4  # For N pumps, this input dimension needs to be 2N.
)

# Now that we have both the system and the controller initialized, we can connect them into a (feedback) Control Loop.
control_loop = cl.ControlLoop(
    raman_system=raman_system,
    controller=controller
)

# A feedback control loop needs to have a target
# The target can be any spectrum that has defined components at same frequencies as the input and outpus spectra of the Raman system in the ControlLoop.
target_powers = ra.Spectrum(ct.Power)
for wl in np.linspace(const.C_BAND[0], const.C_BAND[1], 40):
    target_powers.add_val(conv.wavelength_to_frequency(ct.Length(wl, 'nm')), ct.Power(100, 'uW'))
control_loop.target = target_powers

# We also need to set a control state for the controller
control_loop.curr_control = ra.RamanInputs(
    powers=[ct.Power(200, 'mW'), ct.Power(200, 'mW')],
    wavelengths=[ct.Length(1420, 'nm'), ct.Length(1450, 'nm')]
)

# Finally when we have a target, we can make a full ControlLoop step, by calling the .step() method.
control_loop.step()
# This step method gets the current output of the system, gets the control signal, updates the controllers that need updating, and applies the control signal.
# It is also possible to call methods that perform these steps individually, but do that at your own caution.

# If you want to know an on-off gain of a Raman system, the control loop allows that by the following steps.
off_power = control_loop.off_power_spectrum  # First getting the off power spectrum
assert control_loop.curr_output is not None
on_power = control_loop.curr_output  # Then getting the current output spectrum

on_off_gain_spectrum = on_power/off_power  # And dividing the two spectrum objects

plt.figure()
plt.plot(   
    [conv.frequency_to_wavelenth(f).nm for f in control_loop.curr_output.frequencies],
    [v.dB for v in on_off_gain_spectrum.values],
)
plt.savefig('plot.png')
