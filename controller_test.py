import numpy as np
import logging

import controllers as ctl
import raman_amplifier as ra
import custom_types as ct
import custom_types.conversions as conv


def main():
    log = logging.getLogger("Controller Test Script")

    controller = ctl.PidController()
    ra_input = ra.RamanInputs([ct.Power(0.0, 'mW')], [ct.Length(1450, 'nm')])

    wavelengths = list(np.linspace(1450, 1600, 40))

    def make_flat_spectrum(value: ct.Power, wavelengths: list[float]) -> ra.Spectrum[ct.Power]:
        spectrum = ra.Spectrum(ct.Power)
        for wl in wavelengths:
            power = value
            spectrum.add_val(conv.wavelenth_to_frequency(ct.Length(wl, 'nm')), power)

        return spectrum


    curr_spectrum = make_flat_spectrum(ct.Power(0.1, 'mW'), wavelengths)
    target_spectrum = make_flat_spectrum(ct.Power(0.2, 'mW'), wavelengths)

    for _ in range(100):
        ra_input = controller.get_control(ra_input, curr_spectrum, target_spectrum)
        log.info(ra_input.powers)
