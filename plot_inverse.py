import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

INVERSE_PATH = Path("results/inverse_eval_sigma_0.200.npz")
BERNOULLI_PATH = Path("results/inverse_plus_bern_sigma_0.200.npz")
GRADIENT_PATH = Path("results/better_gd/inverse_plus_gd_sigma_0.200.npz")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_error_pdfs(errors_dict: dict[str, np.ndarray]):
    fig, ax = plt.subplots(figsize=(10, 8))

    # global x-range for fair comparison
    all_errors = np.concatenate(list(errors_dict.values()))
    x = np.linspace(all_errors.min(), all_errors.max(), 500)

    for label, errors in errors_dict.items():
        kde = gaussian_kde(errors)
        pdf = kde(x)

        ax.plot(
            x,
            pdf,
            linewidth=2,
            label=label
        )

    ax.set_xlabel("Mean Gain Error (dB)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Error Distributions (PDF)")
    ax.set_xlim(0, 2)
    ax.grid()
    ax.legend()

    plt.tight_layout()
    plt.show()


def main():
    paths = {
        "inverse": INVERSE_PATH,
        "bern": BERNOULLI_PATH,
        "gradient": GRADIENT_PATH,
    }

    errors_dict = {}

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"No data found at {path}")

        data = np.load(path, allow_pickle=True)
        errors = data["errors"]

        print(f"[{name}] N   = {len(errors)}")
        print(f"[{name}] μ   = {errors.mean():.3f} dB")
        print(f"[{name}] σ   = {errors.std():.3f} dB\n")

        errors_dict[name] = errors

    plot_error_pdfs(errors_dict)


if __name__ == "__main__":
    main()
