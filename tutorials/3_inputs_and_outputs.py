import custom_types as ct
import custom_types.conversions as conv
import raman_amplifier as ra


# Raman inputs represent the pump powers and wavelengths used in the Raman amplifier.
# Although not necessarily amplifier inputs, they are the paramters that are subject to optimization in this control problem formulation.
raman_inputs = ra.RamanInputs(
    powers=[ct.Power(100, 'mW'), ct.Power(50, 'mW')],
    wavelengths=[ct.Length(1450, 'nm'), ct.Length(1470, 'nm')],
)

# Display the Raman inputs
print(raman_inputs)

# Spectrum is an object that represents a spectrum of a signal.
# Here, we create an empty power spectrum.
power_spectrum = ra.Spectrum(
    value_cls=ct.Power,
)

# And here we create a power gain spectrum with pre-populated frequencies and values.
gain_spectrum = ra.Spectrum(
    value_cls=ct.PowerGain,
    frequencies=[ct.Frequency(10, 'THz'), ct.Frequency(11, 'THz')],
    values=[ct.PowerGain(10, 'dB'), ct.PowerGain(0, 'dB')]
)

# It is possible to add values to the spectrum at specific frequencies.
power_spectrum.add_val(
    frequency=ct.Frequency(100, 'MHz'),
    value=ct.Power(100, 'W')
)

# Or at specific wavelegths.
power_spectrum.add_val(
    frequency=conv.wavelength_to_frequency(ct.Length(1550, 'nm')),
    value=ct.Power(100, 'uW')
)

# While expressive, the type system is not defensive.
power_spectrum.add_val(
    frequency=ct.Frequency(100, 'MHz'),
    value=ct.PowerGain(10, 'dB')
)

# And print all of the components in a specturm.
print(power_spectrum)
