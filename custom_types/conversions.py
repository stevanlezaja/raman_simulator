import numpy as np

from .length import Length
from .frequency import Frequency
from .constants import *


def wavelength_to_frequency(wav: Length) -> Frequency:
    return Frequency(LIGHT_SPEED / wav.m, 'Hz')

def frequency_to_wavelenth(freq: Frequency) -> Length:
    return Length(LIGHT_SPEED / freq.Hz, 'm')

def linear_to_db_power(linear_value: float) -> float:
    return 10 * np.log10(linear_value)

def db_to_linear_power(db_value: float) -> float:
    return 10 ** (db_value / 10)
