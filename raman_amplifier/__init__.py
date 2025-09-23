"""
Module containing the Raman Amplifier class and its input-output types
"""

from .raman_amplifier import RamanAmplifier, Pump
from .io import RamanInputs, Spectrum

__all__ = ["RamanAmplifier", "Pump", "RamanInputs", "Spectrum"]
