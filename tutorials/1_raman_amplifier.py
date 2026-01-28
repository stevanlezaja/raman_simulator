import custom_types as ct
import raman_amplifier as ra


# First we'll cover initializing a Raman amplifier object
raman_amplifier = ra.RamanAmplifier(
    num_pumps=2,  # On initialization, you need to specify the number of pumps the amplifier has.
    pumping_ratios=[0, 0]  # Additionally you can specify the individual pumping ratio for each pump, or it will get initialized to 0.5
)

print(raman_amplifier.pump_powers)
print(raman_amplifier.pump_wavelengths)

# You can set pump powers for individual pumps like this:
raman_amplifier.pump_powers = [ct.Power(200, 'mW'), ct.Power(100, 'mW')]

# Individual pump wavelengths like this:
raman_amplifier.pump_wavelengths = [ct.Length(1420, 'nm'), ct.Length(1450, 'nm')]

print(raman_amplifier.pump_powers)
print(raman_amplifier.pump_wavelengths)

# And individual pumping ratios like this:
raman_amplifier.pumping_ratios = [1, 1]
