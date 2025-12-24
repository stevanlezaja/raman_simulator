from pathlib import Path
import json
from copy import deepcopy

import raman_amplifier as ra
from utils.loading_data_from_file import load_raman_dataset


def sort_raman_inputs_inplace(raman_inputs: ra.RamanInputs):
    """
    Sort wavelengths ascending and reorder pumps accordingly.
    Modifies raman_inputs in place.
    """
    pairs = list(zip(raman_inputs.wavelengths, raman_inputs.powers))
    pairs.sort(key=lambda x: x[0])

    sorted_wavelengths, sorted_powers = zip(*pairs)

    raman_inputs.wavelengths[:] = list(sorted_wavelengths)
    raman_inputs.powers[:] = list(sorted_powers)

    return raman_inputs


def write_sorted_dataset_copy(input_path: str, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f_out:
        for raman_inputs, outputs in load_raman_dataset(input_path):
            raman_inputs_sorted = deepcopy(raman_inputs)
            sort_raman_inputs_inplace(raman_inputs_sorted)

            sample = {
                "inputs": raman_inputs_sorted.to_dict(),
                "spectrum": outputs.to_dict(),  # MUST match generator key
            }

            json.dump(sample, f_out)
            f_out.write("\n")

    print(f"Sorted dataset written to: {output_path}")
    print(f"Original dataset unchanged: {input_path}")



if __name__ == '__main__':
    INPUT = "data/raman_simulator/3_pumps/100_fiber_0.0_ratio.json"
    OUTPUT = "data/raman_simulator/3_pumps/100_fiber_0.0_ratio_sorted.json"

    write_sorted_dataset_copy(INPUT, OUTPUT)
