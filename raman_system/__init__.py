"""
Package module exposing the RamanSystem class.

This module imports the RamanSystem from the raman_system submodule
and defines `__all__` to control the public API.

Attributes
----------
RamanSystem : class
    Main class representing a Raman amplification system.
"""

from .raman_system import RamanSystem
from .raman_system_cli import RamanSystemCli

__all__ = ["RamanSystem", "RamanSystemCli"]
