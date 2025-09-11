from raman_amplifier import RamanAmplifier
from controllers import Controller


def get_raman_output():
    pass

def get_control():
    pass

def apply_control():
    pass


class ControlLoop:
    def __init__(self, raman_amplifier: RamanAmplifier, controller: Controller):
        self.raman_amplifier = raman_amplifier
        self.controller = controller
        self.target = None

    def set_target(self, target):
        self.target = target

    def step(self):
        get_raman_output()
        get_control()
        apply_control()

