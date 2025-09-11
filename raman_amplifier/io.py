from custom_types import PowerGain, Length, Power

class RamanInputs:
    def __init__(self):
        self.wavelengths = list[Length]
        self.powers = list[Power]

class GainSpectrum:
    def __init__(self):
        self.spectrum = list[PowerGain]