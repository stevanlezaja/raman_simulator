from collections import defaultdict
import itertools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import hashlib
import math

import custom_types as ct
import raman_system as rs
import raman_amplifier as ra
import fibers as fib
import controllers as ctrl
from entry_points import spectrum_control


RESULTS_DIR = Path("grid_results")
RESULTS_DIR.mkdir(exist_ok=True)

def key_to_path(key):
    h = hashlib.md5(repr(key).encode()).hexdigest()
    return RESULTS_DIR / f"{h}.npz"


if __name__ == '__main__':
    results = defaultdict(list)

    power_steps = [ct.Power(0.1, 'mW')]
    wl_steps = [ct.Length(0.1, 'nm')]
    betas: list[int|float] = [5e3, 1e4]
    gammas: list[float] = [0.7, 0.99]
    weight_decays: list[float] = [1e-3, 1e-1]
    num_samples_list: list[int] = [1]
    repetitions = 6
    simulation_length = 50

    parameters = [len(power_steps), len(betas), len(gammas), len(weight_decays), len(num_samples_list)]
    time_per_iteration = 1.7

    num_combinations = math.prod(parameters)

    total_time = (
        num_combinations
        * repetitions
        * simulation_length
        * time_per_iteration
    )

    print(f"Estimated total runtime: {total_time:.2f} seconds")
    print(f"≈ {total_time / 3600:.2f} hours")

    for [power_step, wl_step], beta, gamma, weight_decay, num_samples in itertools.product(
        zip(power_steps, wl_steps), betas, gammas, weight_decays, num_samples_list
    ):
        key = (power_step, wl_step, beta, gamma, weight_decay, num_samples)
        path = key_to_path(key)

        # Load existing runs if they exist
        if path.exists():
            data = np.load(path, allow_pickle=True)
            runs = list(data["runs"])
            print(f"Loaded {len(runs)} runs with parameters {key}")
        else:
            print(f"\n\nStarting {repetitions} runs with parameters {key}\n")
            runs = []

        while len(runs) < repetitions:
            raman_system = rs.RamanSystem()
            raman_system.fiber = fib.StandardSingleModeFiber(ct.Length(100, "km"))
            raman_system.raman_amplifier = ra.RamanAmplifier(
                num_pumps=3, pumping_ratios=[0, 0, 0]
            )

            controller = ctrl.BernoulliController(
                lr=1e-1,
                power_step=power_step,
                wavelength_step=wl_step,
                beta=beta,
                gamma=gamma,
                weight_decay=weight_decay,
                input_dim=6,
                num_samples=num_samples,
            )

            control_loop = spectrum_control.main(
                iterations=simulation_length,
                controller=controller,
                raman_system=raman_system,
                target_gain_value=np.random.randint(low=5, high=13)
            )

            errors = np.asarray(control_loop.history["errors"])
            if errors.shape[0] < simulation_length:
                last = errors[-1]
                pad_len = simulation_length - errors.shape[0]
                errors = np.concatenate([errors, np.full(pad_len, last)])
            assert errors.shape == (simulation_length,)

            runs.append(errors)

            np.savez(
                path,
                runs=np.stack(runs, axis=0),
                key=repr(key),
            )

    summary = {}

    for file in RESULTS_DIR.glob("*.npz"):
        data = np.load(file, allow_pickle=True)
        runs = data["runs"]      # shape: (10, 100)
        key  = data["key"].item()

        mean = runs.mean(axis=0) # (100,)
        std  = runs.std(axis=0)  # (100,)

        summary[file.stem] = {
            "key": key,
            "mean": mean,
            "std": std,
        }
    
    PLOTS_DIR = Path("grid_plots")
    PLOTS_DIR.mkdir(exist_ok=True)

    for name, item in summary.items():
        mean = item["mean"]
        std  = item["std"]
        key  = item["key"]

        data = np.load(RESULTS_DIR / f"{name}.npz", allow_pickle=True)
        runs = data["runs"]  # (n_runs, n_iter)

        it = np.arange(len(mean))

        plt.figure(figsize=(7, 4))

        for r in runs:
            # plot run
            plt.plot(it, r, color="gray", alpha=0.25, linewidth=1)

            # mark minimum
            idx_min = np.argmin(r)
            plt.scatter(
                it[idx_min],
                r[idx_min],
                color="black",
                s=20,
                zorder=5
            )

        # mean + std
        plt.plot(it, mean, linewidth=2, label="Mean")
        plt.grid()
        plt.fill_between(
            it,
            mean - std,
            mean + std,
            alpha=0.3,
            label="±1 std"
        )

        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.title(key)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{name}.png", dpi=200)
        plt.close()
