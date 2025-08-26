from .length import Length
from .frequency import Frequency
from .constants import *


def wavelenth_to_frequency(wav: Length) -> Frequency:
    return Frequency(LIGHT_SPEED / wav.m, 'Hz')

def frequency_to_wavelenth(freq: Frequency) -> Length:
    return Length(LIGHT_SPEED / freq.Hz, 'm')