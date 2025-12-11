import numpy as np
import pytest

import custom_types as ct
import raman_amplifier as ra


def make_sample_inputs():
    """Helper that returns a known physical RamanInputs object."""
    return ra.RamanInputs(
        powers=[
            ct.Power(0.1, "W"),
            ct.Power(0.5, "W"),
            ct.Power(0.9, "W"),
        ],
        wavelengths=[
            ct.Length(1430, "nm"),
            ct.Length(1450, "nm"),
            ct.Length(1475, "nm"),
        ]
    )


# def test_normalize_values_between_0_and_1():
#     ri = make_sample_inputs()
#     ri.normalize()

#     # All powers/wavelengths must be within [0, 1]
#     for p in ri.powers:
#         assert 0.0 <= p.value <= 1.0

#     for w in ri.wavelengths:
#         assert 0.0 <= w.value <= 1.0


# def test_denormalize_returns_physical_values():
#     ri = make_sample_inputs()
#     ri.normalize()
#     ri.denormalize()

#     # Check ranges are correct
#     p_min, p_max = ra.RamanInputs.power_range
#     wl_min, wl_max = ra.RamanInputs.wavelength_range

#     for p in ri.powers:
#         assert p_min.value <= p.value <= p_max.value

#     for w in ri.wavelengths:
#         assert wl_min.value <= w.value <= wl_max.value


def test_normalize_then_denormalize_roundtrip():
    """Test that denormalize(normalize(x)) returns approximately x."""
    ri = make_sample_inputs()
    arr_original = ri.as_array()

    ri.normalize()
    ri.denormalize()

    arr_roundtrip = ri.as_array()

    # Allow tiny numerical noise from float arithmetic
    assert np.allclose(arr_original, arr_roundtrip, atol=1e-9)


def test_normalization_of_bounds():
    """Explicitly test that min and max map to 0 and 1."""
    p_min, p_max = ra.RamanInputs.power_range
    wl_min, wl_max = ra.RamanInputs.wavelength_range

    ri = ra.RamanInputs(
        powers=[p_min, p_max],
        wavelengths=[wl_min, wl_max],
    )

    ri.normalize()

    assert pytest.approx(ri.powers[0].value, rel=0, abs=1e-12) == 0.0
    assert pytest.approx(ri.powers[1].value, rel=0, abs=1e-12) == 1.0
    assert pytest.approx(ri.wavelengths[0].value, rel=0, abs=1e-12) == 0.0
    assert pytest.approx(ri.wavelengths[1].value, rel=0, abs=1e-12) == 1.0
