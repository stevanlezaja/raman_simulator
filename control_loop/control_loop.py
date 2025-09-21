from typing import Optional
import logging

import raman_amplifier as ra
import experiment as exp
import controllers as ctrl
import custom_types as ct
import custom_types.conversions as conv


log = logging.Logger("[Control Loop]", level=logging.INFO)


class ControlLoop:
    def __init__(self, raman_system: exp.Experiment, controller: ctrl.Controller):
        self.raman_system = raman_system
        self.controller = controller
        self.target: Optional[ra.Spectrum[ct.Power]] = None
        self.curr_control: ra.RamanInputs = ra.RamanInputs(n_pumps=1)
        self.curr_output: Optional[ra.Spectrum[ct.Power]] = None

    def set_target(self, target: ra.Spectrum[ct.Power]):
        self.target = target

    def get_raman_output(self) -> ra.Spectrum[ct.Power]:
        log.debug("Current pump power:", self.raman_system.raman_amplifier.pump_power)
        self.raman_system.update()
        output_power = self.raman_system.get_signal_power_at_distance(self.raman_system.fiber.length)
        log.debug("Current output power:", output_power)
        output_frequency = conv.wavelenth_to_frequency(self.raman_system.signal.wavelength)
        output = ra.Spectrum(ct.Power)
        output.add_val(output_frequency, output_power)
        return output

    def get_control(self) -> ra.RamanInputs:
        assert self.curr_output is not None
        if self.target is None:
            return ra.RamanInputs()
        control = self.controller.get_control(curr_input=self.curr_control, curr_output=self.curr_output, target_output=self.target)
        return control

    def apply_control(self):
        log.debug("Old pump power: ", self.raman_system.raman_amplifier.pump_power)
        self.raman_system.raman_amplifier.pump_power += self.curr_control.powers[0]
        log.debug("New pump power: ", self.raman_system.raman_amplifier.pump_power)
        self.raman_system.raman_amplifier.pump_wavelength += self.curr_control.wavelengths[0]

    def step(self):
        self.curr_output = self.get_raman_output()
        self.curr_control = self.get_control()
        self.apply_control()

