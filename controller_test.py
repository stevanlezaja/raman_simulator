import numpy as np
import controllers as ctl
import raman_amplifier as ra
import custom_types as ct
import custom_types.conversions as conv
import custom_logging as clog


def main():
    log = clog.get_logger("Controller Test Script")

    # Use gradient descent controller
    controller = ctl.GradientDescentController()
    if isinstance(controller, ctl.GradientDescentController):
        controller.train_controller('data/raman_simulator_3_pumps_1.0_ratio.json')

    # Initialize RamanInputs with 3 pumps (typical for your setup)
    n_pumps = 3
    ra_input = ra.RamanInputs(
        powers=[ct.Power(0.0, 'mW') for _ in range(n_pumps)],
        wavelengths=[ct.Length(1450 + 10*i, 'nm') for i in range(n_pumps)]
    )

    wavelengths = list(np.linspace(1450, 1600, 40))

    def make_flat_spectrum(value: ct.Power, wavelengths: list[float]) -> ra.Spectrum[ct.Power]:
        spectrum = ra.Spectrum(ct.Power)
        for wl in wavelengths:
            freq = conv.wavelength_to_frequency(ct.Length(wl, 'nm'))
            spectrum.add_val(freq, value)
        return spectrum

    curr_spectrum = make_flat_spectrum(ct.Power(0.1, 'mW'), wavelengths)
    target_spectrum = make_flat_spectrum(ct.Power(0.2, 'mW'), wavelengths)

    # Run 100 iterations of gradient descent control
    for step in range(100):
        ra_input = controller.get_control(ra_input, curr_spectrum, target_spectrum)
        log.info(f"Step {step+1}: {', '.join(f'{p.mW:.3f} mW' for p in ra_input.powers)}")


if __name__ == "__main__":
    clog.setup_logging()
    main()
