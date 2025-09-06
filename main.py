import numpy as np
import matplotlib.pyplot as plt

from fibers import Fiber, DispersionCompensatingFiber, SuperLargeEffectiveArea, StandardSingleModeFiber, NonZeroDispersionFiber
from signals import Signal
from raman_amplifier import RamanAmplifier
from experiment.experiment import Experiment
from utils import to_dB, from_dB
from custom_types import Length, Power


def plot_Pp_Ps_over_distance():
    sig = Signal()
    fib = DispersionCompensatingFiber()
    ra = RamanAmplifier()
    ra.pump_wavelength = Length(1555, 'nm')
    ra.pump_power = Power(1, 'mW')
    exp = Experiment(fib, sig, ra)

    distances = np.linspace(0, fib.length.km, 1000)

    pump_powers = []
    signal_powers = []
    for dist in distances:
        pump_powers.append(to_dB(exp.get_pump_power_at_distance(Length(dist, 'km')), exp.get_pump_power_at_distance(Length(0, 'km'))))
        signal_powers.append(to_dB(exp.get_signal_power_at_distance(Length(dist, 'km')), exp.get_signal_power_at_distance(Length(0, 'km'))))

    plt.figure()
    plt.plot(distances, pump_powers, label="Pump power")
    plt.plot(distances, signal_powers, label="Signal power")
    plt.legend()
    plt.show()

def calculate_G_net():
    sig = Signal()
    fib = DispersionCompensatingFiber()
    ra = RamanAmplifier()
    exp = Experiment(fib, sig, ra)

    G_net = exp.get_signal_power_at_distance(fib.length) / exp.get_pump_power_at_distance(Length(0, 'm'))
    print("G_net = ", G_net)

def caluclate_G_on_off():
    sig = Signal()
    fib = SuperLargeEffectiveArea()
    fib.length.km = 10.0
    fib.alpha_s = 0.0437 * 1e-3
    fib.alpha_p = 0.0576 * 1e-3
    ra = RamanAmplifier()
    ra.pump_power.W = 1.24
    exp = Experiment(fib, sig, ra)
    exp.solve()

    G_on = exp.get_signal_power_at_distance(fib.length)

    ra.pump_power.W = 0.0
    exp.solve()

    G_off = exp.get_signal_power_at_distance(fib.length)

    G_on_off = G_on / G_off

    print("G_on_off = ", to_dB(G_on_off), "dB")

def plot_raman_efficiencies():
    fibers = [NonZeroDispersionFiber(), DispersionCompensatingFiber(), StandardSingleModeFiber(), SuperLargeEffectiveArea()]
    for fiber in fibers:
        fig = fiber.make_raman_efficiency_figure()
        fig.show()

def main():
    # plot_Pp_Ps_over_distance()
    # calculate_G_net()
    # caluclate_G_on_off()
    plot_raman_efficiencies()

if __name__ == "__main__":
    # from runner.run import Runner

    # runner = Runner()

    # runner.run()
    main()