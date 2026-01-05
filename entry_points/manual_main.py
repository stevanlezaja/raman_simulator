import custom_types as ct
import controllers as ctrl
import fibers as fib
import raman_amplifier as ra
import entry_points.spectrum_control as spectrum_control
import raman_system as rs
from utils import parser


def main():
    main_parser = parser.get_main_parser()
    main_args = main_parser.parse_args()

    kwargs = {k: v for k, v in vars(main_args).items() if v is not None}

    _ = kwargs.pop('customize', False)

    n_pumps = 3
    pumping_ratios = [0.0 for _ in range(n_pumps)]

    raman_system = rs.RamanSystem()
    raman_system.fiber = fib.StandardSingleModeFiber()
    raman_system.fiber.length.km = 100
    raman_system.raman_amplifier = ra.RamanAmplifier(num_pumps=n_pumps, pumping_ratios=pumping_ratios)

    controller = ctrl.ManualController(
        n_pumps=n_pumps,
        power_step=ct.Power(5, 'mW'),
        wavelength_step=ct.Length(10, 'nm')
    )

    spectrum_control.main(**kwargs, raman_system=raman_system, controller=controller, iterations=100, target_gain_value=7)
