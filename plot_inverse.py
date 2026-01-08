import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CHECKPOINT_PATH = Path("results/inverse_model_eval_checkpoint.npz")


def plot_error_distribution(errors: np.ndarray):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))  # type: ignore

    # PDF
    ax.hist(errors, bins=40, density=True, alpha=0.7)  # type: ignore
    ax.set_xlabel("Mean Gain Error (dB)")  # type: ignore
    ax.set_ylabel("Probability Density")  # type: ignore
    ax.set_title(f"Error Distribution (PDF), N={len(errors)}")  # type: ignore
    ax.grid()  # type: ignore

    plt.tight_layout()
    plt.show()  # type: ignore


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"No data found at {CHECKPOINT_PATH}")

    data = np.load(CHECKPOINT_PATH, allow_pickle=True)
    errors = data["errors"]

    print(f"Loaded {len(errors)} samples")
    print(f"Mean error: {errors.mean():.3f} dB")
    print(f"Std error : {errors.std():.3f} dB")

    plot_error_distribution(errors)


if __name__ == "__main__":
    main()
