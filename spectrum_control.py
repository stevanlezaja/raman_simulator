import numpy as np
import matplotlib.pyplot as plt

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

    target_spectrum[target_spectrum.frequencies[30]].mW = 200

    raman_system.input_spectrum = input_spectrum
    raman_system.output_spectrum = input_spectrum

    # controller = ctrl.BernoulliController()
    controller = ctrl.PidController(p=1, i=0, d=0)
    control_loop = loop.ControlLoop(raman_system, controller)
    control_loop.set_target(target_spectrum)

    # store errors
    errors = []

    # enable interactive plotting
    plt.ion()

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for step in range(1000):
        print(f"\n\n STEP: {step}")
        control_loop.step()
        assert control_loop.curr_output is not None and control_loop.target is not None
        error = control_loop.curr_output - control_loop.target
        errors.append(error.mean)

        # --- update error over time plot ---
        ax1.clear()
        ax1.plot(errors, label="Error mean")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Error (mean)")
        ax1.set_title("Error over time")
        ax1.legend()
        fig1.canvas.draw()
        fig1.canvas.flush_events()

        # --- update target vs output spectra plot ---
        ax2.clear()
        ax2.plot(
            list([f.Hz for f in control_loop.target.frequencies]),
            [val.value for val in control_loop.target.values],
            label="Target",
        )
        ax2.plot(
            list([f.Hz for f in control_loop.curr_output.frequencies]),
            [val.value for val in control_loop.curr_output.values],
            label="Current Output",
        )
        ax2.set_xlabel("Frequency (Hz)")  # or convert back to nm if you prefer
        ax2.set_ylabel("Power (mW)")
        ax2.set_title("Target vs. Current Output")
        ax2.legend()
        fig2.canvas.draw()
        fig2.canvas.flush_events()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
