import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from tqdm import tqdm

import custom_types as ct
import custom_types.constants as const
import custom_types.conversions as conv
import control_loop as cl
import raman_system as rs
import raman_amplifier as ra
import fibers as fib
import controllers as ctrl
from utils.loading_data_from_file import load_raman_dataset
from pathlib import Path


CHECKPOINT_PATH = Path("results/inverse_model_eval_checkpoint.npz")
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_checkpoint(path: Path) -> list[float]:
    if path.exists():
        data = np.load(path, allow_pickle=True)
        errors = data["errors"].tolist()
        print(f"Resuming from checkpoint: {len(errors)} samples completed")
        return errors
    return []


def save_checkpoint(path: Path, errors: list[float]):
    tmp_path = path.with_suffix(".tmp.npz")
    np.savez(tmp_path, errors=np.array(errors))
    tmp_path.replace(path)


def mean_gain_error_db(
    loop: cl.ControlLoop,
    predicted_spectrum: ra.Spectrum[ct.Power],
    real_spectrum: ra.Spectrum[ct.Power],
) -> float:
    pred_gain = predicted_spectrum / loop.off_power_spectrum
    real_gain = real_spectrum / loop.off_power_spectrum

    pred_db = np.array([v.dB for v in pred_gain.values])
    real_db = np.array([v.dB for v in real_gain.values])

    return float(np.mean(np.abs(pred_db - real_db)))


def bar_plot_raman_inputs(ax: Axes, raman_inputs: ra.RamanInputs, predicted_inputs: ra.RamanInputs):  # type: ignore
    x = np.arange(6)
    width = 0.25
    ax.bar(x - width, raman_inputs.normalize().as_array(), width, label="Target")  # type: ignore
    ax.bar(x, predicted_inputs.normalize().as_array(), width, label="RPM")  # type: ignore


def plot_spectrums(ax: Axes, loop: cl.ControlLoop, predicted_spectrum: ra.Spectrum[ct.Power], real_spectrum: ra.Spectrum[ct.Power]):
    predicted_gain = predicted_spectrum/loop.off_power_spectrum
    real_gain = real_spectrum/loop.off_power_spectrum
    ax.plot( # type: ignore
        [f.Hz for f in predicted_gain.frequencies],
        [val.dB for val in predicted_gain.values],
        label="Predicted",
    )
    ax.plot( # type: ignore
        [f.Hz for f in real_gain.frequencies],
        [val.dB for val in real_gain.values],
        label="Simulated",
    )

    ymax = max(val.dB for val in predicted_gain.values)
    ymax = max(ymax, max(val.dB for val in real_gain.values))

    ax.set_xlabel("Frequency (Hz)")  # type: ignore
    ax.set_ylabel("Gain (dB)")  # type: ignore
    ax.set_ylim(0, 1.05 * ymax)
    ax.set_title("Target vs Current Output Spectrum")  # type: ignore
    ax.grid()  # type: ignore
    ax.legend()  # type: ignore


def plot_error_distribution(errors: np.ndarray):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))  # type: ignore

    # PDF (histogram)
    ax[0].hist(errors, bins=40, density=True, alpha=0.7)
    ax[0].set_xlabel("Mean Gain Error (dB)")  # type: ignore
    ax[0].set_ylabel("Probability Density")  # type: ignore
    ax[0].set_title("Error Distribution (PDF)")  # type: ignore
    ax[0].grid()  # type: ignore

    # CDF
    sorted_err = np.sort(errors)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)

    ax[1].plot(sorted_err, cdf)
    ax[1].set_xlabel("Mean Gain Error (dB)")  # type: ignore
    ax[1].set_ylabel("CDF")  # type: ignore
    ax[1].set_title("Cumulative Distribution Function")  # type: ignore
    ax[1].grid()  # type: ignore

    plt.tight_layout()
    plt.show()  # type: ignore



def main():
    raman_system = rs.RamanSystem()
    raman_system.fiber = fib.StandardSingleModeFiber(ct.Length(100, 'km'))
    raman_system.raman_amplifier = ra.RamanAmplifier(3, [0.0, 0.0, 0.0])

    input_spectrum = ra.Spectrum(ct.Power)
    for num in np.linspace(const.C_BAND[0], const.C_BAND[1], 40):
        freq = conv.wavelength_to_frequency(ct.Length(num, 'nm'))
        input_spectrum.add_val(freq, ct.Power(25, 'uW'))

    raman_system.input_spectrum = input_spectrum
    raman_system.output_spectrum = copy.deepcopy(input_spectrum)

    controller = ctrl.PidController(0, 0, 0)
    loop = cl.ControlLoop(raman_system, controller)

    # 🔹 Load checkpoint
    errors: list[float] = load_checkpoint(CHECKPOINT_PATH)
    start_idx = len(errors)

    dataset = list(load_raman_dataset(
        'data/raman_simulator/3_pumps/100_fiber_0.0_ratio_sorted.json'
    ))

    for i in tqdm(range(start_idx, len(dataset))):
        raman_inputs, spectrum = dataset[i]

        predicted_inputs = loop.inverse_model.get_raman_inputs(spectrum)
        loop.curr_control = predicted_inputs
        loop.apply_control()
        predicted_spectrum = copy.deepcopy(loop.get_raman_output())

        err = mean_gain_error_db(loop, predicted_spectrum, spectrum)
        errors.append(err)

        # 🔹 Save every N samples
        if i % 10 == 0:
            save_checkpoint(CHECKPOINT_PATH, errors)

    # Final save
    save_checkpoint(CHECKPOINT_PATH, errors)

    errors_np = np.array(errors)
    print(f"\nMean error: {errors_np.mean():.3f} dB")
    print(f"Std error : {errors_np.std():.3f} dB")

    plot_error_distribution(errors_np)


if __name__ == '__main__':
    main()