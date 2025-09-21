import control_loop as loop
import raman_amplifier as ra
import fibers as fib
import signals as sig
import experiment as exp
import custom_types as ct
import custom_types.conversions as conv
import controllers as ctrl
import custom_logging as clog


def main():
    log = clog.get_logger("Control Loop Test Script")

    fiber = fib.StandardSingleModeFiber()
    fiber.length.km = 10

    signal = sig.Signal()
    signal.power.mW = 1
    signal.wavelength.nm = 1550

    raman_amplifier = ra.RamanAmplifier(0.5)

    raman_system = exp.RamanSystem(fiber, signal, raman_amplifier)

    controller = ctrl.PidController(p=1.0, i=0.0, d=0.0)

    control_loop = loop.ControlLoop(raman_system, controller)

    target = ra.Spectrum(ct.Power)
    target.add_val(conv.wavelenth_to_frequency(signal.wavelength), ct.Power(15, 'mW'))

    control_loop.set_target(target)

    for i in range(10):
        control_loop.step()
        if control_loop.curr_output is not None:
            log.info(f"STEP: {i}")
            log.info(f"Control: {control_loop.curr_control.powers}")
            log.info(f"Spectrum: {control_loop.curr_control.powers}")
