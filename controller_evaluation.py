import copy
import numpy as np
from tqdm import tqdm
from pathlib import Path

import custom_types as ct
import custom_types.constants as const
import custom_types.conversions as conv
import control_loop as cl
import raman_system as rs
import raman_amplifier as ra
import fibers as fib
import controllers as ctrl
import models as m
from utils.loading_data_from_file import load_raman_dataset
from entry_points import spectrum_control


# =========================
# Configuration
# =========================

DATASET_PATH = "data/raman_simulator/3_pumps/100_fiber_0.0_ratio_sorted.json"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GD_ITERATIONS = 50
CHECKPOINT_EVERY = 1


# =========================
# Utilities
# =========================

def load_checkpoint(path: Path) -> list[float]:
    if path.exists():
        data = np.load(path, allow_pickle=True)
        errors = data["errors"].tolist()
        print(f"Resuming from checkpoint: {len(errors)} samples")
        return errors
    return []


def save_checkpoint(path: Path, errors: list[float]):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, errors=np.array(errors))
    tmp.replace(path)


def mean_gain_error_db(
    loop: cl.ControlLoop,
    predicted_spectrum: ra.Spectrum[ct.Power],
    target_spectrum: ra.Spectrum[ct.Power],
) -> float:
    pred_gain = predicted_spectrum / loop.off_power_spectrum
    target_gain = target_spectrum / loop.off_power_spectrum

    pred_db = np.array([v.dB for v in pred_gain.values])
    targ_db = np.array([v.dB for v in target_gain.values])

    return float(np.mean(np.abs(pred_db - targ_db)))


def main():
    for sigma_rpm in [0.2]:
        checkpoint_path = RESULTS_DIR / f"inverse_plus_gd_sigma_{sigma_rpm:.3f}.npz"
        errors = load_checkpoint(checkpoint_path)
        start_idx = len(errors)
        for i in tqdm(range(start_idx, 200)):
            raman_system = rs.RamanSystem()
            raman_system.fiber = fib.StandardSingleModeFiber(ct.Length(100, "km"))
            raman_system.raman_amplifier = ra.RamanAmplifier(3, [0.0, 0.0, 0.0])

            input_spectrum = ra.Spectrum(ct.Power)
            for num in np.linspace(const.C_BAND[0], const.C_BAND[1], 40):
                freq = conv.wavelength_to_frequency(ct.Length(num, "nm"))
                input_spectrum.add_val(freq, ct.Power(25, "uW"))

            raman_system.input_spectrum = input_spectrum
            raman_system.output_spectrum = copy.deepcopy(input_spectrum)

            gd_controller = ctrl.GradientDescentController(
                training_data='controllers/gradient_descent_controller/data/raman_simulator_3_pumps_0.0_ratio.json',
                epochs=5000,
                lr_control=1e-1,
                iterations=100,
            )

            bern_controller = ctrl.BernoulliController(  # type: ignore
                lr=1e-1,
                power_step=ct.Power(0.1, 'mW'),
                wavelength_step=ct.Length(0.1, 'nm'),
                beta=10000,
                gamma=0.99,
                weight_decay=1e-1,
                sigma=1,
                input_dim=6,
            )
            loop = cl.ControlLoop(raman_system, gd_controller)

            dataset = list(load_raman_dataset(DATASET_PATH))

            print(f"\n=== Evaluating Inverse + GD (sigma={sigma_rpm}) ===")

            loop.inverse_model = m.InverseModel(sigma_rpm=sigma_rpm)

            # for i in tqdm(range(start_idx, len(dataset))):
            raman_inputs, target_spectrum = dataset[i]  # type: ignore

            predicted_inputs = loop.inverse_model.get_raman_inputs(target_spectrum)
            loop.curr_control = predicted_inputs
            loop.apply_control()

            loop.set_target(target_spectrum)

            spectrum_control.main(
                save_plots=False,
                iterations=GD_ITERATIONS,
                control_loop=loop,
            )

            final_spectrum = copy.deepcopy(loop.curr_output)
            assert final_spectrum is not None

            err = mean_gain_error_db(loop, loop.curr_output, target_spectrum)
            print(err)
            errors.append(err)

            if i % CHECKPOINT_EVERY == 0:
                save_checkpoint(checkpoint_path, errors)

        save_checkpoint(checkpoint_path, errors)

        errors_np = np.array(errors)
        print(f"Mean error: {errors_np.mean():.3f} dB")
        print(f"Std error : {errors_np.std():.3f} dB")


if __name__ == "__main__":
    main()
