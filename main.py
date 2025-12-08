import numpy as np
import matplotlib.pyplot as plt

from fibers import Fiber, DispersionCompensatingFiber, SuperLargeEffectiveArea, \
                StandardSingleModeFiber, NonZeroDispersionFiber, ChristopheExperimentFiber
from signals import Signal
from raman_amplifier import RamanAmplifier
from experiment.experiment import Experiment
from utils.utils import to_dB, from_dB
from custom_types import Length, Power


def plot_Pp_Ps_over_distance(exp: Experiment):
    distances = np.linspace(0, exp.fiber.length.km, 1000)
    pump_powers = []
    signal_powers = []
    for dist in distances:
        # pump_powers.append(to_dB(exp.get_pump_power_at_distance(Length(dist, 'km')), exp.get_pump_power_at_distance(Length(0, 'km'))))
        pump_powers.append(exp.get_pump_power_at_distance(Length(dist, 'km')) / exp.get_pump_power_at_distance(Length(0, 'km')))
        # signal_powers.append(to_dB(exp.get_signal_power_at_distance(Length(dist, 'km')), exp.get_signal_power_at_distance(Length(0, 'km'))))
        signal_powers.append(exp.get_signal_power_at_distance(Length(dist, 'km')) / exp.get_signal_power_at_distance(Length(0, 'km')))

    plt.figure()
    plt.plot(distances, pump_powers, label="Pump power")
    plt.plot(distances, signal_powers, label="Signal power")
    plt.grid()
    plt.legend()
    plt.show()

def plot_Ps_over_fiber_length(exp: Experiment):
    distances = np.linspace(0, exp.fiber.length.km, 1000)
    signal_powers = []
    for dist in distances:
        signal_powers.append(to_dB(exp.get_signal_power_at_distance(Length(dist, 'km')), Power(1, 'mW').W))

    plt.figure()
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
    fig, ax = plt.subplots()
    for fiber in fibers:
        fiber: Fiber
        ax = fiber.plot_raman_efficiency(ax)
    fig.show()

def net_gain_experiment():
    """
        From Bromage 2004 p82
    """
    signal = Signal()
    signal.wavelength.nm = 1555

    fiber = SuperLargeEffectiveArea()
    fiber.length.km = 10.0
    fiber.alpha_s.m = 0.0437 * 1e-3
    fiber.alpha_p.m = 0.0576 * 1e-3

    raman_amplifier = RamanAmplifier(pumping_ratio=1)
    raman_amplifier.pump_power = Power(1.24, 'W')

    experiment_on = Experiment(fiber, signal, raman_amplifier)
    power_on = experiment_on.get_signal_power_at_distance(experiment_on.fiber.length)

    raman_amplifier.pump_power = Power(0.0, 'W')
    experiment_off = Experiment(fiber, signal, raman_amplifier)
    power_off = experiment_off.get_signal_power_at_distance(experiment_off.fiber.length)

    G_net = power_on / power_off

    print("G_net = ", to_dB(G_net), "dB")

def validation_experiment(fiber: Fiber, ax):
    """"
        Experiment provided by Christophe to validate the model for Raman amplification
    """
    fiber.length.km = 100
    fiber.alpha_p.dB_km = 0.3
    fiber.alpha_s.dB_km = 0.2

    signal = Signal()
    signal.wavelength.nm = 1550
    signal.power.mW = 0.1

    raman_amplifier = RamanAmplifier()
    raman_amplifier.pump_wavelength.nm = 1450
    raman_amplifier.pump_power = Power(758.2, 'mW')

    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

    for ratio in ratios:
        raman_amplifier.pumping_ratio = ratio
        exp = Experiment(fiber, signal, raman_amplifier)

        distances = np.linspace(0, exp.fiber.length.km, 100)
        signal_powers = []
        for dist in distances:
            signal_powers.append(to_dB(exp.get_signal_power_at_distance(Length(dist, 'km')).W, Power(1, 'mW').W))
        ax.plot(distances, signal_powers, label=f"r = {ratio}")


def main():
    # plot_Pp_Ps_over_distance(Experiment(DispersionCompensatingFiber(), Signal(), RamanAmplifier()))
    # calculate_G_net()
    # caluclate_G_on_off()
    # plot_raman_efficiencies()
    # net_gain_experiment()
    fibers: list[fib.Fiber] = [StandardSingleModeFiber(), NonZeroDispersionFiber(), SuperLargeEffectiveArea()]
    fig, axes = plt.subplots(2, 2)  # type: ignore
    axes = axes.flatten()
    for fiber, ax in zip(fibers, axes):
        validation_experiment(fiber, ax)
        ax.set_xlabel('Distance [km]')
        ax.set_ylabel('Signal power [dBm]')
        ax.legend()
        ax.grid()
        ax.set_title(fiber.__name__)
    plt.show()  # type: ignore


if __name__ == "__main__":
    import spectrum_control
    import raman_system as rs
    import controllers as ctrl
    from utils import parser

    main_parser = parser.get_main_parser()
    main_args = main_parser.parse_args()

    kwargs = {k: v for k, v in vars(main_args).items() if v is not None}

    customize = kwargs.pop('customize', False)

    if customize:
        raman_system_cli = rs.RamanSystemCli()
        raman_system = raman_system_cli.make()
        controller_cli = ctrl.ControllerCli()
        controller = controller_cli.make()

        spectrum_control.main(**kwargs, raman_system=raman_system, controller=controller)
    else:
        import fibers as fib
        import raman_amplifier as ra
        import controllers as ctrl
        import custom_types as ct

        raman_system = rs.RamanSystem()
        raman_system.fiber = fib.StandardSingleModeFiber()
        raman_system.fiber.length.km = 80
        raman_system.raman_amplifier = ra.RamanAmplifier(num_pumps=3, pumping_ratios=[0, 0, 0])

        # controller = ctrl.BernoulliController(
        #     lr=1e-1,
        #     power_step=ct.Power(20, 'mW'),
        #     wavelength_step=ct.Length(2, 'nm'),
        #     beta=1000,
        #     gamma=0.99,
        #     weight_decay=1e-1,
        #     input_dim=6,
        # )

        # controller = ctrl.DifferentialEvolutionController()

        controller = ctrl.GradientDescentController(training_data='controllers/gradient_descent_controller/data/raman_simulator_3_pumps_1.0_ratio.json', epochs=250)

        spectrum_control.main(**kwargs, raman_system=raman_system, controller=controller, iterations=100)
