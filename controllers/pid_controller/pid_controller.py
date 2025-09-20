import raman_amplifier as ra
import custom_types as ct

from ..controller_base import Controller

class PidController(Controller):
    def __init__(self, p:float=0.2, i:float=0.1, d:float=0.1):
        self.p = p
        self.i = i
        self.d = d
        self.integral = 0.0
        self.e1 = 0.0

    def get_control(self, curr_input: ra.RamanInputs, curr_output: ra.Spectrum[ct.Power], target_output: ra.Spectrum[ct.Power]) -> ra.RamanInputs:
        e = (target_output - curr_output).mean

        p = self.p * e

        i = self.integral + self.i * e
        if abs(i) < 1:
            self.integral = i

        d = self.d * (e - self.e1)
        self.e1 = e

        control = p + i + d

        delta_powers = [ct.Power(control, 'W') for _ in curr_input.powers]
        delta_wavelengths = [wl for wl in curr_input.wavelengths]

        return ra.RamanInputs(powers=delta_powers, wavelengths=delta_wavelengths)


if __name__ == "__main__":
    pid = PidController()

    initial = ra.Spectrum(ct.Power)
    initial.spectrum = {
        ct.Frequency(0, 'Hz'): ct.Power(10, ' '),
        ct.Frequency(1, 'Hz'): ct.Power(15, ' '),
        ct.Frequency(2, 'Hz'): ct.Power(20, ' '),
        ct.Frequency(3, 'Hz'): ct.Power(25, ' '),
        ct.Frequency(4, 'Hz'): ct.Power(7, ' '),
    }

    target = ra.Spectrum(ct.Power)
    target.spectrum = {
        ct.Frequency(0, 'Hz'): ct.Power(1, ' '),
        ct.Frequency(1, 'Hz'): ct.Power(2, ' '),
        ct.Frequency(2, 'Hz'): ct.Power(3, ' '),
        ct.Frequency(3, 'Hz'): ct.Power(4, ' '),
        ct.Frequency(4, 'Hz'): ct.Power(1, ' '),
    }

    curr_input = ra.RamanInputs([ct.Power(0.0, 'W')], [ct.Length(0.0, 'm')])


    for i in range(100):
        curr_input = pid.get_control(curr_input, initial, target)
        print(curr_input.powers)