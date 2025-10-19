"""
controllers.bernoulli_controller package
========================================

This package provides the BernoulliController for Raman Amplifier control.

The `BernoulliController` implements a stochastic, Bernoulli-based
reinforcement learning policy to adjust pump powers and wavelengths
based on observed gain spectra.

Modules
-------
bernoulli_controller
    Contains the BernoulliController class and its implementation.

Exports
-------
BernoulliController
    The main controller class available for external use.
"""

from .bernoulli_controler import BernoulliController

__all__ = ['BernoulliController']
