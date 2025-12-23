import controllers as ctrl
import custom_types as ct
import fibers as fib
import raman_amplifier as ra
import entry_points.spectrum_control as spectrum_control
import raman_system as rs
from utils import parser


def main():

    controller = ctrl.BernoulliController(
        lr=1e-1,
        power_step=ct.Power(20, 'mW'),
        wavelength_step=ct.Length(2, 'nm'),
        beta=1000,
        gamma=0.99,
        weight_decay=1e-1,
        input_dim=6,
    )

    main_parser = parser.get_main_parser()
    main_args = main_parser.parse_args()

    kwargs = {k: v for k, v in vars(main_args).items() if v is not None}

    _ = kwargs.pop('customize', False)

    raman_system = rs.RamanSystem()
    raman_system.fiber = fib.StandardSingleModeFiber()
    raman_system.fiber.length.km = 100
    raman_system.raman_amplifier = ra.RamanAmplifier(num_pumps=3, pumping_ratios=[0, 0, 0])

    spectrum_control.main(**kwargs, raman_system=raman_system, controller=controller)
