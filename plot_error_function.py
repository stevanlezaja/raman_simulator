import copy

import numpy as np
import matplotlib.pyplot as plt

import custom_types as ct
import custom_types.conversions as conv

import raman_system as rs
import raman_amplifier as ra
import fibers as fib


LOWER = 1500
UPPER = 1600
SAMPLES = 40


def main(
        num_samples: int = SAMPLES,
        wavelength_range: tuple[float, float] = (LOWER, UPPER),
        raman_system: rs.RamanSystem = rs.RamanSystem(),
) -> None:

    input_spectrum = ra.Spectrum(ct.Power)
    for num in list(np.linspace(wavelength_range[0], wavelength_range[1], num_samples)):
        freq = conv.wavelenth_to_frequency(ct.Length(num, 'nm'))
        input_spectrum.add_val(freq, ct.Power(10, 'mW'))

    target_spectrum = ra.Spectrum(ct.Power)
    # create values
    dummy_sytem = rs.RamanSystem()
    dummy_sytem.raman_amplifier = ra.RamanAmplifier()
    dummy_sytem.fiber = copy.deepcopy(raman_system.fiber)
    dummy_sytem.input_spectrum = copy.deepcopy(input_spectrum)
    dummy_sytem.output_spectrum = copy.deepcopy(input_spectrum)
    dummy_sytem.raman_amplifier.pump_power.mW = 500
    dummy_sytem.raman_amplifier.pump_wavelength.nm = 1450

    dummy_sytem.update()

    target_spectrum = dummy_sytem.output_spectrum

    raman_system.input_spectrum = input_spectrum
    raman_system.output_spectrum = copy.deepcopy(input_spectrum)

    powers: list[ct.Power] = []
    for num in list(np.linspace(0, 1, 25)):
        powers.append(ct.Power(num, 'W'))

    wavelengths: list [ct.Length] = []
    for num in list(np.linspace(1420, 1490, 25)):
        wavelengths.append(ct.Length(num, 'nm'))


    errors = np.zeros((len(powers), len(wavelengths)))
    
    for i, p in enumerate(powers):
        for j, w in enumerate(wavelengths):
            raman_system.raman_amplifier.pump_power = p
            raman_system.raman_amplifier.pump_wavelength = w
            raman_system.update()

            error_spectrum = target_spectrum - raman_system.output_spectrum
            # Example: mean absolute error or squared error
            error_val = error_spectrum.mean ** 0.5
            errors[i, j] = error_val

    powers_W = np.array([p.W for p in powers])
    wavelengths_nm = np.array([w.nm for w in wavelengths])

    mid_power_idx = len(powers_W) // 2
    mid_wave_idx = len(wavelengths_nm) // 2

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Error Analysis: Pump Power vs Wavelength', fontsize=14)

    # --- (1) Main 2D error grid ---
    im = axes[0, 0].imshow(
        errors.T,
        extent=[powers_W.min(), powers_W.max(), wavelengths_nm.min(), wavelengths_nm.max()],
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )
    axes[0, 0].set_title('Error vs Power & Wavelength')
    axes[0, 0].set_xlabel('Pump Power [W]')
    axes[0, 0].set_ylabel('Pump Wavelength [nm]')
    fig.colorbar(im, ax=axes[0, 0], label='Mean Error')

    # --- (2) Error vs wavelength (power fixed at mid) ---
    axes[0, 1].plot(wavelengths_nm, errors[mid_power_idx, :], marker='o')
    axes[0, 1].set_title(f'Error vs Wavelength\n(Power = {powers_W[mid_power_idx]:.2f} W)')
    axes[0, 1].set_xlabel('Pump Wavelength [nm]')
    axes[0, 1].set_ylabel('Mean Error')

    # --- (3) Error vs power (wavelength fixed at mid) ---
    axes[1, 0].plot(powers_W, errors[:, mid_wave_idx], marker='o', color='orange')
    axes[1, 0].set_title(f'Error vs Power\n(Wavelength = {wavelengths_nm[mid_wave_idx]:.1f} nm)')
    axes[1, 0].set_xlabel('Pump Power [W]')
    axes[1, 0].set_ylabel('Mean Error')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    raman_system = rs.RamanSystem()
    raman_system.raman_amplifier = ra.RamanAmplifier()
    raman_system.fiber = fib.StandardSingleModeFiber()
    main(raman_system=raman_system)
