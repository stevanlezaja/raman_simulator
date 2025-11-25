"""
Module containing tests for io.py
"""

import pytest

import custom_types as ct

import raman_amplifier as ra

def test_ramaninputs_default_init():
    """
    Default initialization with n_pumps only
    """
    n_pumps = 3
    inp = ra.RamanInputs(n_pumps=n_pumps)
    assert len(inp.powers) == n_pumps
    assert len(inp.wavelengths) == n_pumps
    assert all(p.value == 0.0 for p in inp.powers)
    assert all(w.value == 0.0 for w in inp.wavelengths)

def test_ramaninputs_custom_values():
    """
    Initialization with powers and wavelengths
    """
    powers = [ct.Power(1.0, 'W'), ct.Power(2.0, 'W')]
    wavelengths = [ct.Length(1.5, 'm'), ct.Length(2.0, 'm')]
    inp = ra.RamanInputs(powers=powers, wavelengths=wavelengths)
    assert inp.powers == powers
    assert inp.wavelengths == wavelengths
    # Check value_dict
    for w, p in zip(wavelengths, powers):
        assert inp.value_dict[w] == p

def test_ramaninputs_partial_init_raises():
    """
    Error on partial initialization
    """
    powers = [ct.Power(1.0, 'W')]
    with pytest.raises(AssertionError):
        ra.RamanInputs(powers=powers)

def test_spectrum_creation():
    """
    Creation
    """
    spec = ra.Spectrum(ct.Power)
    assert spec.value_cls == ct.Power
    assert not spec.spectrum

def test_spectrum_add_val():
    """
    Adding values
    """
    spec = ra.Spectrum(ct.Power)
    f1 = ct.Frequency(1.0, 'THz')
    p1 = ct.Power(0.5, 'W')
    spec.add_val(f1, p1)
    assert spec.spectrum[f1] == p1
    assert spec.frequencies == [f1]
    assert spec.values == [p1]

def test_spectrum_add_sub():
    """
    Linear operations (+, -)
    """
    spec1 = ra.Spectrum(ct.Power)
    spec2 = ra.Spectrum(ct.Power)
    f1 = ct.Frequency(1.0, 'THz')
    spec1.add_val(f1, ct.Power(1.0, 'W'))
    spec2.add_val(f1, ct.Power(2.0, 'W'))

    result = spec1 + spec2
    assert result.spectrum[f1].value == 3.0

    result2 = spec2 - spec1
    assert result2.spectrum[f1].value == 1.0

def test_spectrum_mean():
    """
    Mean property
    """
    spec = ra.Spectrum(ct.Power)
    spec.add_val(ct.Frequency(1.0, 'THz'), ct.Power(1.0, 'W'))
    spec.add_val(ct.Frequency(2.0, 'THz'), ct.Power(3.0, 'W'))
    assert spec.mean == (1 ** 2 + 3 ** 2) ** 0.5 / 2

def test_spectrum_mismatched_keys():
    """
    Mismatched frequencies
    """
    spec1 = ra.Spectrum(ct.Power)
    spec2 = ra.Spectrum(ct.Power)
    spec1.add_val(ct.Frequency(1.0, 'THz'), ct.Power(1.0, 'W'))
    spec2.add_val(ct.Frequency(2.0, 'THz'), ct.Power(1.0, 'W'))
    with pytest.raises(AssertionError):
        _ = spec1 + spec2
