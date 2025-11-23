import pytest
import custom_types as ct
import raman_amplifier as ra


# ----------------------------
# RamanInputs Serialization
# ----------------------------

def test_raman_inputs_to_dict():
    inp = ra.RamanInputs(
        powers=[ct.Power(100, "mW"), ct.Power(200, "mW")],
        wavelengths=[ct.Length(1450, "nm"), ct.Length(1460, "nm")]
    )

    d = inp.to_dict()

    assert d["powers_mW"] == [100.0, 200.0]
    assert d["wavelengths_nm"] == [1450.0, 1460.0]


def test_raman_inputs_round_trip():
    original = ra.RamanInputs(
        powers=[ct.Power(123, "mW"), ct.Power(456, "mW")],
        wavelengths=[ct.Length(1423, "nm"), ct.Length(1475, "nm")]
    )

    d = original.to_dict()
    restored = ra.RamanInputs.from_dict(d)

    assert len(restored.powers) == 2
    assert len(restored.wavelengths) == 2

    assert restored.powers[0].mW == pytest.approx(123.0)
    assert restored.powers[1].mW == pytest.approx(456.0)

    assert restored.wavelengths[0].nm == pytest.approx(1423.0)
    assert restored.wavelengths[1].nm == pytest.approx(1475.0)


# ----------------------------
# Spectrum Serialization
# ----------------------------

def test_spectrum_power_round_trip():
    spec = ra.Spectrum(ct.Power)
    freqs = [ct.Frequency(1e14, "Hz"), ct.Frequency(2e14, "Hz")]
    vals = [ct.Power(1.23, "mW"), ct.Power(4.56, "mW")]

    for f, v in zip(freqs, vals):
        spec.add_val(f, v)

    d = spec.to_dict()
    restored = ra.Spectrum(ct.Power).from_dict(d)

    assert len(restored.frequencies) == 2
    assert len(restored.values) == 2

    assert restored.frequencies[0].Hz == pytest.approx(1e14)
    assert restored.values[0].mW == pytest.approx(1.23)


def test_spectrum_powergain_round_trip():
    spec = ra.Spectrum(ct.PowerGain)
    freqs = [ct.Frequency(1e14, "Hz"), ct.Frequency(2e14, "Hz")]
    vals = [ct.PowerGain(3.2, "dB"), ct.PowerGain(5.5, "dB")]

    for f, v in zip(freqs, vals):
        spec.add_val(f, v)

    d = spec.to_dict()
    restored = ra.Spectrum(ct.PowerGain).from_dict(d)

    assert restored.values[0].dB == pytest.approx(3.2)
    assert restored.values[1].dB == pytest.approx(5.5)


# ----------------------------
# Failure mode: missing keys
# ----------------------------

def test_spectrum_missing_values_raises():
    bad_dict = {
        "frequencies_Hz": [1e14, 2e14],
        # missing values_mW or values_dB
    }

    with pytest.raises(TypeError):
        ra.Spectrum(ct.Power).from_dict(bad_dict)


def test_inputs_missing_key_raises():
    bad_dict = {
        "powers_mW": [100, 200]
        # missing wavelengths_nm
    }

    with pytest.raises(KeyError):
        ra.RamanInputs.from_dict(bad_dict)
