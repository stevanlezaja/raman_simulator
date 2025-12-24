import json
from pathlib import Path
from typing import Iterator, Iterable

import custom_types as ct
import raman_amplifier as ra


def _load_jsonl(path: str) -> Iterator[dict]:
    """
    Loads the dataset file line-by-line.
    Each line is a separate JSON dict.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # skip empty lines
                yield json.loads(line)


def load_raman_dataset(path: str) -> Iterable[tuple[ra.RamanInputs, ra.Spectrum[ct.Power]]]:
    """
    Loads the dataset and reconstructs RamanInputs and Spectrum objects.
    """
    for entry in _load_jsonl(path):

        inputs_dict = entry["inputs"]
        spectrum_dict = entry["spectrum"]

        # Convert back to objects
        raman_inputs = ra.RamanInputs.from_dict(inputs_dict)
        spectrum = ra.Spectrum(ct.Power).from_dict(spectrum_dict)

        yield raman_inputs, spectrum


def main():
    dataset_path = "data/raman_simulator_3_pumps_1.0_ratio.json"

    print(f"Loading dataset from {dataset_path}")

    for i, (inp, spec) in enumerate(load_raman_dataset(dataset_path), 1):
        print(f"\n--- Sample #{i} ---")
        print("RamanInputs:", inp)
        print("Spectrum first 5 points:", list(spec.values)[:5])

        if i >= 3:
            break  # stop early for preview


if __name__ == "__main__":
    main()
