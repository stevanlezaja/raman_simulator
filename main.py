import numpy as np
import matplotlib.pyplot as plt

from fibers import DispersionCompensatingFiber, SuperLargeEffectiveArea
from signals import Signal
from raman_amplifier import RamanAmplifier
from utils import to_dB, from_dB

def plot_Pp_Ps_over_distance():
    sig = Signal()
    fib = DispersionCompensatingFiber()
    ra = RamanAmplifier(fib, sig)

    distances = np.linspace(0, fib.length, 100)

    pump_power = ra.get_pump_power_at_distance(distances)
    signal_power = ra.get_signal_power_at_distance(distances)

    plt.figure()
    plt.plot(distances, pump_power, label="Pump power")
    plt.plot(distances, signal_power, label="Signal power")
    plt.legend()
    plt.show()

def calculate_G_net():
    sig = Signal()
    fib = DispersionCompensatingFiber()
    ra = RamanAmplifier(fib, sig)

    G_net = ra.get_signal_power_at_distance(fib.length) / ra.get_pump_power_at_distance(0)
    print("G_net = ", G_net)

def caluclate_G_on_off():
    sig = Signal()
    fib = SuperLargeEffectiveArea()
    fib.length = 10 * 1e3
    fib.alpha_s = 0.0437 * 1e-3
    fib.alpha_p = 0.0576 * 1e-3
    ra = RamanAmplifier(fib, sig)
    ra.pump_power = 1.24
    ra.solve()

    G_on = ra.get_signal_power_at_distance(fib.length)

    ra.pump_power = 0.0
    ra.solve()

    G_off = ra.get_signal_power_at_distance(fib.length)

    G_on_off = G_on / G_off

    print("G_on_off = ", to_dB(G_on_off), "dB")


def main():
    # plot_Pp_Ps_over_distance()
    calculate_G_net()
    caluclate_G_on_off()

if __name__ == "__main__":
    main()