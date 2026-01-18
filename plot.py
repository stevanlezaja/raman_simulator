import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


RESULTS_DIR = Path("grid_results")
RESULTS_DIR.mkdir(exist_ok=True)

summary = {}

for file in RESULTS_DIR.glob("*.npz"):
    data = np.load(file, allow_pickle=True)
    runs = data["runs"]
    key  = data["key"].item()

    mean = runs.mean(axis=0)
    std  = runs.std(axis=0)

    start_error = np.mean(runs[:, 0])
    final_error = np.mean(runs[:, -1])
    best_error = np.mean(runs.min(axis=1))
    
    print("Parameters:", key)
    print("Final error", final_error - start_error)
    print("Best error ", best_error - start_error, "\n")

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
    runs = data["runs"]

    it = np.arange(len(mean))

    plt.figure(figsize=(7, 4))

    for r in runs:
        plt.plot(it, r - r[0] * np.ones_like(r), color="gray", alpha=0.25, linewidth=1)

        idx_min = np.argmin(r)
        plt.scatter(
            it[idx_min],
            r[idx_min] - r[0],
            color="black",
            s=20,
            zorder=5
        )

    plt.plot(it, mean - mean[0] * np.ones_like(r), linewidth=2, label="Mean")

    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.grid()
    plt.title("Loss Evolution with Gradient Descent Controller")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{name}.png", dpi=200)
    plt.close()
