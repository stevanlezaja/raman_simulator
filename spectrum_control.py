import numpy as np

import custom_logging as clog
import custom_types as ct
import custom_types.conversions as conv

import control_loop as loop
import raman_system as rs
import fibers as fib
import raman_amplifier as ra
import controllers as ctrl


log = clog.get_logger("Spectrum Control Test Script")


def main() -> None:

    fiber = fib.StandardSingleModeFiber()
    fiber.length.km = 10

    raman_system = rs.RamanSystem()
    raman_system.raman_amplifier = ra.RamanAmplifier()
    raman_system.fiber = fiber

    input_spectrum = ra.Spectrum(ct.Power)
    for num in list(np.linspace(1450, 1600, 40)):
        freq = conv.wavelenth_to_frequency(ct.Length(num, 'nm'))
        input_spectrum.add_val(freq, ct.Power(1, 'mW'))

    target_spectrum = ra.Spectrum(ct.Power)
    for num in list(np.linspace(1450, 1600, 40)):
        freq = conv.wavelenth_to_frequency(ct.Length(num, 'nm'))
        target_spectrum.add_val(freq, ct.Power(10, 'mW'))

    raman_system.input_spectrum = input_spectrum
    raman_system.output_spectrum = input_spectrum

    controller = ctrl.BernoulliController()

    control_loop = loop.ControlLoop(raman_system, controller)
    control_loop.set_target(target_spectrum)

    for _ in range(100):
        control_loop.step()
        assert control_loop.curr_output is not None and control_loop.target is not None
        error = control_loop.curr_output - control_loop.target
        print(error.mean)

if __name__ == "__main__":
    main()
