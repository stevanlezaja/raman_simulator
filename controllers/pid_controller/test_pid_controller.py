"""
Module containing tests for PID Controller class
"""

import custom_types as ct
import raman_amplifier as ra

from .pid_controller import PidController


def test_initialization():
    """"
    Testing initialization of PidController class objects
    """
    p_val = 1
    i_val = 1.4
    d_val = 0.7

    pid = PidController(p_val, i_val, d_val)

    assert pid.p == p_val
    assert pid.i == i_val
    assert pid.d == d_val
    assert pid.e1 == 0
    assert pid.integral == 0


def test_p_control():
    """
    Testing the control strategy of a P controller
    """
    pid = PidController(p = 1, i = 0, d = 0)

    initial = ra.Spectrum(ct.Power)
    initial.spectrum = {
        ct.Frequency(0, 'Hz'): ct.Power(1, ' '),
    }

    target = ra.Spectrum(ct.Power)
    target.spectrum = {
        ct.Frequency(0, 'Hz'): ct.Power(2, ' '),
    }

    curr_input = ra.RamanInputs([ct.Power(0.0, 'W')], [ct.Length(0.0, 'm')])

    curr_input = pid.get_control(curr_input, initial, target)
    assert curr_input.powers[0].W == 1

    assert curr_input.wavelengths[0].m == 0


def test_i_control():
    """
    Testing the control strategy of an I controller
    """
    i_val = 1
    pid = PidController(p = 0, i = i_val, d = 0)

    initial = ra.Spectrum(ct.Power)
    initial.spectrum = {
        ct.Frequency(0, 'Hz'): ct.Power(1, ' '),
    }

    target = ra.Spectrum(ct.Power)
    target.spectrum = {
        ct.Frequency(0, 'Hz'): ct.Power(2, ' '),
    }

    error = (initial - target).mean

    curr_input = ra.RamanInputs([ct.Power(0.0, 'W')], [ct.Length(0.0, 'm')])

    curr_input = pid.get_control(curr_input, initial, target)
    assert pid.integral == -error * i_val
    assert curr_input.powers[0].W == -error * i_val

    curr_input = pid.get_control(curr_input, initial, target)

    assert curr_input.powers[0].W == 2
    assert pid.integral == 2


def test_d_control():
    """
    Testing the control strategy of a D controller
    """
    d_val = 1
    pid = PidController(p = 0, i = 0, d = d_val)

    initial = ra.Spectrum(ct.Power)
    initial.spectrum = {
        ct.Frequency(0, 'Hz'): ct.Power(1, ' '),
    }

    target = ra.Spectrum(ct.Power)
    target.spectrum = {
        ct.Frequency(0, 'Hz'): ct.Power(2, ' '),
    }

    error = (initial - target).mean

    curr_input = ra.RamanInputs([ct.Power(0.0, 'W')], [ct.Length(0.0, 'm')])

    curr_input = pid.get_control(curr_input, initial, target)
    assert pid.e1 == -error
    assert curr_input.powers[0].W == -error * d_val

    curr_input = pid.get_control(curr_input, initial, target)
    assert pid.e1 == -error
    assert curr_input.powers[0].W == (pid.e1 + error) * d_val


def test_output_format():
    """
    Testing the output format of a PID controller
    """
    pid = PidController()

    initial = ra.Spectrum(ct.Power)
    initial.spectrum = {
        ct.Frequency(0, 'Hz'): ct.Power(1, ' '),
    }

    initial_input = ra.RamanInputs([ct.Power(0.0, 'W')], [ct.Length(0.0, 'm')])

    curr_input = pid.get_control(initial_input, initial, initial)

    assert len(curr_input.powers) == len(initial_input.powers)
    assert len(curr_input.wavelengths) == len(initial_input.wavelengths)
