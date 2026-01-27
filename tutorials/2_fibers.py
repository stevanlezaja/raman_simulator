import matplotlib.pyplot as plt

import custom_types as ct
import fibers as fib


# First we'll cover initializing a fiber object
fiber = fib.StandardSingleModeFiber(
    length=ct.Length(100, 'km'),  # On initialization, you can specify the length of the fiber.
)

# You can also set the length of the fiber after initialization like this:
fiber.length = ct.Length(50, 'km')

# The fiber object has separate pump and signal frequency dependent attenuation attributes:
print("Fiber attenuation at pump frequency in default units:",fiber.alpha_p)
print("Fiber attenuation at pump frequency in dB/km:", fiber.alpha_p.dB_km, "dB/km")
print("Fiber attenuation at signal frequency in default units:", fiber.alpha_s)
print("Fiber effective area:", fiber.effective_area)

# You can see the fiber's raman gain efficiency like this:
fig, ax = plt.subplots(1, 1)
fiber.plot_raman_efficiency(ax=ax)
plt.savefig("fiber_raman_efficiency.png")
