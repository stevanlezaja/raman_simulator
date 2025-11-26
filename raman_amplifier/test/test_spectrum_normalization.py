import numpy as np
import pytest
import raman_amplifier as ra
import custom_types as ct

def make_sample_spectrum():
    freqs = [ct.Frequency(1e9 + i * 1e6, 'Hz') for i in range(10)]
    values = [ct.Power(float(i + 1), 'W') for i in range(10)]
    return ra.Spectrum(ct.Power, frequencies=freqs, values=values)

def test_spectrum_normalize_denormalize_round_trip():
    # set class-wide min and max
    ra.Spectrum.norm_min = 1.0
    ra.Spectrum.norm_max = 10.0

    s = make_sample_spectrum()
    s.normalize()
    vals_norm = [v.value for v in s.values]
    assert min(vals_norm) == 0.0
    assert max(vals_norm) == 1.0

    s.denormalize()
    vals_denorm = [v.value for v in s.values]
    for i, val in enumerate(vals_denorm):
        assert np.isclose(val, float(i + 1))

def test_normalized_values_in_unit_interval():
    ra.Spectrum.norm_min = 1.0
    ra.Spectrum.norm_max = 10.0

    s = make_sample_spectrum()
    s.normalize()
    vals = [v.value for v in s.values]
    assert all(0.0 <= v <= 1.0 for v in vals)

def test_denormalization_restores_original_values():
    ra.Spectrum.norm_min = 1.0
    ra.Spectrum.norm_max = 10.0

    s = make_sample_spectrum()
    original_values = [v.value for v in s.values]
    s.normalize()
    s.denormalize()
    restored_values = [v.value for v in s.values]
    for orig, restored in zip(original_values, restored_values):
        assert np.isclose(orig, restored)

def test_constant_spectrum_raises_error():
    # all values identical, class-wide min/max must allow normalization
    ra.Spectrum.norm_min = 5.0
    ra.Spectrum.norm_max = 5.0

    freqs = [ct.Frequency(i, 'Hz') for i in range(5)]
    values = [ct.Power(5.0, 'W') for _ in range(5)]
    s = ra.Spectrum(ct.Power, frequencies=freqs, values=values)
    with pytest.raises(ValueError):
        s.normalize()

def test_negative_values_normalization():
    ra.Spectrum.norm_min = -10.0
    ra.Spectrum.norm_max = 10.0

    freqs = [ct.Frequency(i, 'Hz') for i in range(5)]
    values = [ct.Power(v, 'W') for v in np.linspace(-10, 10, 5)]
    s = ra.Spectrum(ct.Power, frequencies=freqs, values=values)
    s.normalize()
    vals = [v.value for v in s.values]
    assert all(0.0 <= v <= 1.0 for v in vals)
