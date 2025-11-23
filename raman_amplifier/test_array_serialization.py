from typing import Any
import numpy as np
import pytest

import custom_types as ct
import raman_amplifier as ra


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def assert_roundtrip(cls: ra.RamanInputs|ra.Spectrum[Any], obj: ra.RamanInputs|ra.Spectrum[Any]):
    """Test obj → array → obj → array."""
    arr1 = obj.as_array()
    if cls == ra.RamanInputs:
        obj2 = cls.from_array(arr1)
    elif cls == ra.Spectrum:
        obj2 = cls.from_array(ct.Power, arr1)
    else:
        raise TypeError
    arr2 = obj2.as_array()
    assert np.allclose(arr1, arr2), f"Round-trip failed for {cls.__name__}"


def random_ra_inputs():
    """Generate valid random ra.RamanInputs object."""
    powers = [ct.Power(float(np.random.rand()), "W") for _ in range(3)]
    wavelengths = [
        ct.Length(float(np.random.uniform(1420, 1500)), "nm")
        for _ in range(3)
    ]
    return ra.RamanInputs(powers=powers, wavelengths=wavelengths)


def random_gain_spectrum():
    """Generate valid random ra.Spectrum object."""
    return ra.Spectrum.from_array(ct.Power, np.random.randn(40))


# ------------------------------------------------------------
# ra.RamanInputs tests
# ------------------------------------------------------------

def test_ra_inputs_basic_roundtrip():
    obj = ra.RamanInputs(
        powers=[
            ct.Power(0.1, "W"),
            ct.Power(0.2, "W"),
            ct.Power(0.3, "W"),
        ],
        wavelengths=[
            ct.Length(1450, "nm"),
            ct.Length(1470, "nm"),
            ct.Length(1490, "nm"),
        ]
    )
    assert_roundtrip(ra.RamanInputs, obj)


def test_ra_inputs_shape_validation():
    wrong = np.zeros(5)  # should be 6
    with pytest.raises((ValueError, AssertionError)):
        ra.RamanInputs.from_array(wrong)


def test_ra_inputs_random_fuzz():
    """Test 200 random valid round-trips."""
    for _ in range(200):
        obj = random_ra_inputs()
        assert_roundtrip(ra.RamanInputs, obj)


def test_ra_inputs_no_aliasing():
    """Ensure as_array does NOT return internal mutable references."""
    obj = random_ra_inputs()
    arr = obj.as_array()
    arr_copy = arr.copy()
    arr[:] = 999  # mutate returned array
    # For correctness, original object should not change
    assert np.allclose(obj.as_array(), arr_copy)


# ------------------------------------------------------------
# ra.Spectrum tests
# ------------------------------------------------------------

def test_gain_spectrum_basic_roundtrip():
    gs = ra.Spectrum.from_array(ct.Power, np.linspace(0, 10, 40))
    assert_roundtrip(ra.Spectrum, gs)


def test_gain_spectrum_shape_validation():
    wrong = np.zeros(39)  # should be 40
    with pytest.raises((ValueError, AssertionError)):
        ra.Spectrum.from_array(ct.Power, wrong)


def test_gain_spectrum_random_fuzz():
    for _ in range(200):
        gs = random_gain_spectrum()
        assert_roundtrip(ra.Spectrum, gs)


def test_gain_spectrum_no_aliasing():
    gs = random_gain_spectrum()
    arr = gs.as_array()
    arr_copy = arr.copy()
    arr[:] = 777  # mutate returned array
    assert np.allclose(gs.as_array(), arr_copy)


# ------------------------------------------------------------
# Cross-object tests
# ------------------------------------------------------------

def test_class_identity():
    """Ensure serialization preserves semantically equal objects."""
    obj = random_ra_inputs()
    arr = obj.as_array()
    restored = ra.RamanInputs.from_array(arr)

    # Compare semantic values (powers and wavelengths)
    for p1, p2 in zip(obj.powers, restored.powers):
        assert np.isclose(p1.W, p2.W)

    for l1, l2 in zip(obj.wavelengths, restored.wavelengths):
        assert np.isclose(l1.nm, l2.nm)
