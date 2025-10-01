"""
Package module exposing the ControlLoop class.

This module imports the ControlLoop from the control_loop submodule
and defines `__all__` to control the public API.

Attributes
----------
ControlLoop : class
    Main class implementing a feedback control loop for a Raman amplification system.
"""

from .control_loop import ControlLoop

__all__ = ["ControlLoop"]
