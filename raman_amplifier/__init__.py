"""
Module containing the Raman Amplifier class and its input-output types
"""

from .raman_amplifier import RamanAmplifier, Pump
from .spectrum import Spectrum, mse
from .raman_inputs import RamanInputs
from .raman_amplifier_cli import RamanAmplifierCli

__all__ = ["RamanAmplifier", "Pump", "RamanInputs", "Spectrum", "RamanAmplifierCli",
           "mse"]
