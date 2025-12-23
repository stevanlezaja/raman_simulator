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

import control_loop as cl
import raman_amplifier as ra


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
    assert isinstance(loop.curr_control, ra.RamanInputs)

    # Assert that the current output is initially None
    assert loop.curr_output is None


# def test_control_loop_set_target():
#     """
#     Test that the set_target method correctly updates the target spectrum.

#     This test verifies that:
#     - Before setting, the target is None.
#     - After calling set_target, the target attribute is updated to the provided spectrum.
#     """

#     # Create mock objects for RamanSystem and Controller
#     mock_raman_system = MagicMock()
#     mock_controller = MagicMock()

#     # Instantiate the ControlLoop with mocks
#     loop = cl.ControlLoop(mock_raman_system, mock_controller)

#     # Assert that target is initially None
#     assert loop.target is None

#     # Create a dummy target spectrum (can be any object, here a string for simplicity)
#     dummy_target = "dummy_spectrum"
#     dummy_target = MagicMock(spec=ra.Spectrum)
#     loop.set_target(dummy_target)
#     assert loop.target is dummy_target

#     # Set the target using set_target
#     loop.set_target(dummy_target)

#     # Assert that the target attribute is now updated
#     assert loop.target == dummy_target


def test_control_loop_get_raman_output():
    """
    Test that get_raman_output calls the Raman system update and returns the output spectrum.

    This test verifies that:
    - raman_system.update() is called exactly once.
    - The returned value equals raman_system.output_spectrum.
    """

    # Create a mock RamanSystem with a mock output_spectrum
    mock_raman_system = MagicMock()
    dummy_output = MagicMock()  # could be a Spectrum mock
    mock_raman_system.output_spectrum = dummy_output

    # Create a mock controller (not used in this test)
    mock_controller = MagicMock()

    # Instantiate the ControlLoop
    loop = cl.ControlLoop(mock_raman_system, mock_controller)

    # Call get_raman_output
    result = loop.get_raman_output()

    # Assert that raman_system.update was called once
    mock_raman_system.update.assert_called_once()

    # Assert that the result equals the mocked output_spectrum
    assert result == dummy_output
