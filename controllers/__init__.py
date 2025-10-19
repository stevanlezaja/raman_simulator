"""
controllers package
===================

This package provides a variety of controller implementations for Raman
Amplifier systems, as well as a command-line interface for interactive
controller selection and configuration.

Modules
-------
bernoulli_controller
    Implements the BernoulliController, a stochastic reinforcement learning
    policy for adjusting pump powers and wavelengths.
pid_controller
    Implements a PID-based controller for classical feedback control.
manual_controller
    Implements a ManualController for user-defined, static control inputs.
controller_base
    Defines the abstract base class `Controller` for all controllers.
controller_cli
    Provides `ControllerCli`, a command-line interface for selecting and
    instantiating controllers interactively.

Exports
-------
ControllerCli
    Interactive CLI for controller selection and parameter configuration.
ManualController
    Manual controller implementation.
BernoulliController
    Stochastic RL-based controller implementation.
PidController
    Classical PID controller implementation.
Controller
    Abstract base class for all controllers.
"""

from .bernoulli_controler import BernoulliController
from .pid_controller import PidController
from .controller_base import Controller
from .manual_controller import ManualController
from .controller_cli import ControllerCli

__all__ = ["ControllerCli", "ManualController", "BernoulliController", "PidController",
           "Controller"]
