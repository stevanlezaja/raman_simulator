import random

import raman_amplifier as ra


def test_init():
    num_pumps = 3

    raman_amplifier = ra.RamanAmplifier(num_pumps)

    assert len(raman_amplifier.pump_pairs) == num_pumps
    for pair in raman_amplifier.pump_pairs:
        assert len(pair) == 2


def test_pumping_ratio_init():
    num_pumps = 3

    pumping_ratios: list[float] = []
    for _ in range(num_pumps):
        pumping_ratios.append(random.uniform(0.0, 1.0))

    raman_amplifier = ra.RamanAmplifier(num_pumps, pumping_ratios)

    for i, pair in enumerate(raman_amplifier.pump_pairs):
        forward, backward = pair
        assert forward.power == raman_amplifier.pump_powers[i] * pumping_ratios[i]
        assert backward.power == raman_amplifier.pump_powers[i] * (1 - pumping_ratios[i])
