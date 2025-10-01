"""
Tests for the ControlLoop class in the control_loop module.

This test suite verifies the core functionality of the ControlLoop, which manages
a feedback control loop for a Raman amplification system.

The tests cover:

- Initialization of the ControlLoop instance and default attributes.
- Setting the target spectrum via `set_target`.
- Retrieving Raman system output via `get_raman_output`.
- Computing control inputs via `get_control`.
- Applying control inputs to the Raman amplifier via `apply_control`.
- Performing a full control loop iteration via `step`.
- Edge cases such as missing target or missing output.

Mocks are used for the RamanSystem and Controller objects to isolate the ControlLoop
behavior from actual amplifier simulations or controller logic.
"""

from unittest.mock import MagicMock
import pytest

import control_loop as cl
from raman_amplifier import RamanInputs


def test_control_loop_initialization():
    """
    Test that a ControlLoop instance initializes correctly.

    This test verifies that:
    - The provided Raman system and controller are stored in the loop.
    - The target spectrum is initially None.
    - The current control (`curr_control`) is a RamanInputs instance.
    - The current output (`curr_output`) is initially None.
    """

    # Create mock objects for RamanSystem and Controller
    mock_raman_system = MagicMock()
    mock_controller = MagicMock()

    # Instantiate the ControlLoop with the mocks
    loop = cl.ControlLoop(mock_raman_system, mock_controller)

    # Assert that the loop stores the RamanSystem object correctly
    assert loop.raman_system is mock_raman_system

    # Assert that the loop stores the Controller object correctly
    assert loop.controller is mock_controller

    # Assert that the target is initially None
    assert loop.target is None

    # Assert that the current control is an instance of RamanInputs
    # This ensures the loop has a valid default control object
    assert isinstance(loop.curr_control, RamanInputs)

    # Assert that the current output is initially None
    assert loop.curr_output is None
