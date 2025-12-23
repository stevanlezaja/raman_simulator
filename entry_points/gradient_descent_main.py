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

    raman_system = rs.RamanSystem()
    raman_system.fiber = fib.StandardSingleModeFiber()
    raman_system.fiber.length.km = 100
    raman_system.raman_amplifier = ra.RamanAmplifier(num_pumps=3, pumping_ratios=[0, 0, 0])

    controller = ctrl.GradientDescentController(training_data='controllers/gradient_descent_controller/data/raman_simulator_3_pumps_0.0_ratio.json', epochs=500, lr_control=10)

    spectrum_control.main(**kwargs, raman_system=raman_system, controller=controller, iterations=100)

