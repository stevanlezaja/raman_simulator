import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


RESULTS_DIR = Path("grid_results")
RESULTS_DIR.mkdir(exist_ok=True)

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
        plt.plot(it, r - r[0] * np.ones_like(r), color="gray", alpha=0.25, linewidth=1)

        # mark minimum
        idx_min = np.argmin(r)
        plt.scatter(
            it[idx_min],
            r[idx_min] - r[0],
            color="black",
            s=20,
            zorder=5
        )

    # mean + std
    plt.plot(it, mean - mean[0] * np.ones_like(r), linewidth=2, label="Mean")
    plt.fill_between(
        it,
        mean - std - mean[0] * np.ones_like(r),
        mean + std - mean[0] * np.ones_like(r),
        alpha=0.3,
        label="±1 std"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.grid()
    plt.title(key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{name}.png", dpi=200)
    plt.close()
