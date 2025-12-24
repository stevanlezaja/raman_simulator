import copy
import random
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import custom_types as ct
import custom_types.conversions as conv
import custom_types.constants as c
import raman_amplifier as ra
import raman_system as rs
import fibers as fib
from utils import parser
from data import write_sorted_dataset_copy


def sample_raman_inputs(num_pumps: int, power_range: tuple[ct.Power, ct.Power], wavelength_range: tuple[ct.Length, ct.Length]) -> ra.RamanInputs:
    powers: list[ct.Power] = []
    wavelengths: list[ct.Length] = []
    for _ in range(num_pumps):
        powers.append(ct.Power(random.uniform(round(power_range[0].mW), round(power_range[1].mW)), 'mW'))
        wavelengths.append(ct.Length(random.uniform(round(wavelength_range[0].nm), round(wavelength_range[1].nm)), 'nm'))
    return ra.RamanInputs(powers, wavelengths)


def write_data(data: list[dict[str, Any]], file_path: str) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        for sample in data:
            json.dump(sample, f)
            f.write("\n")

def generate_data(num_samples: int, num_pumps: int, pumping_ratio: float, fib_len: ct.Length, file_path: str, plot: bool = False, *args, **kwds) -> None:
    raman_system = rs.RamanSystem()
    raman_system.raman_amplifier = ra.RamanAmplifier(num_pumps, [pumping_ratio for _ in range(num_pumps)])
    raman_system.fiber = fib.StandardSingleModeFiber(fib_len)
    raman_system.input_spectrum = ra.Spectrum(ct.Power)
    for num in list(np.linspace(c.C_BAND[0], c.C_BAND[1], 40)):
        freq = conv.wavelength_to_frequency(ct.Length(num, 'nm'))
        raman_system.input_spectrum.add_val(freq, ct.Power(25, 'uW'))
    raman_system.output_spectrum = copy.deepcopy(raman_system.input_spectrum)

    power_range = (ct.Power(ra.RamanInputs.MIN_POWER_W, 'W'), ct.Power(ra.RamanInputs.MAX_POWER_W, 'W'))
    wavelength_range = (ct.Length(ra.RamanInputs.MIN_WAVELENGTH_NM, 'nm'), ct.Length(ra.RamanInputs.MAX_WAVELENGTH_NM, 'nm'))

    data_batch: list[dict[str, Any]] = []
    batch_size = 10

    if plot:
        plt.ion()  # type: ignore
        fig, ax = plt.subplots()  # type: ignore

    for _ in tqdm(range(num_samples)):
        raman_inputs = sample_raman_inputs(num_pumps, power_range, wavelength_range)

        # Run system
        raman_system.raman_inputs = raman_inputs
        raman_system.update()
        power_spectrum = copy.deepcopy(raman_system.output_spectrum)

        # Store serialized dict
        data_batch.append({
            "inputs": raman_inputs.to_dict(),
            "spectrum": power_spectrum.to_dict()
        })

        # Write batch
        if len(data_batch) >= batch_size:
            write_data(data_batch, file_path)
            if plot: visualize_data_batch(data_batch, ax)  # type: ignore
            data_batch = []

    # Write remaining
    if data_batch:
        write_data(data_batch, file_path)


def _plot_single_raman_input(raman_inputs: ra.RamanInputs, ax: Any) -> None:
    powers = [p.mW for p in raman_inputs.powers]
    wavelengths = [w.nm for w in raman_inputs.wavelengths]
    ax.scatter(powers, wavelengths, s=10)


def visualize_data_batch(data_batch: list[dict[str, Any]], ax: Any) -> None:
    for item in data_batch:
        raman_inputs = ra.RamanInputs.from_dict(item['inputs'])
        _plot_single_raman_input(raman_inputs, ax)
    plt.pause(0.001)


def visualize_data(file_path: str = 'data/raman_simulator/3_pumps/100_fiber_0.0_ratio_sorted.json', update_every: int = 10):
    from utils.loading_data_from_file import load_raman_dataset

    plt.ion()  # type: ignore
    fig, ax = plt.subplots()  # type: ignore

    ax.set_xlabel("Pump Power (mW)")  # type: ignore
    ax.set_ylabel("Pump Wavelength (nm)")  # type: ignore
    ax.set_title("Streaming Raman Inputs")  # type: ignore
    ax.grid(True)  # type: ignore

    for i, (raman_inputs, _) in enumerate(load_raman_dataset(file_path)):
        _plot_single_raman_input(raman_inputs, ax)

        if i % update_every == 0:
            plt.pause(0.001)

    plt.ioff()  # type: ignore
    plt.show()  # type: ignore


def main():

    gen_parser = parser.data_generator_parser()
    args = gen_parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if v is not None}

    num_pumps = kwargs['num_pumps']
    pumping_ratio = kwargs['pumping_ratio']
    fib_len = ct.Length(kwargs['fiber_length'], 'km')
    file_path = f'data/raman_simulator/{num_pumps}_pumps/{fib_len.km:.0f}_fiber_{pumping_ratio}_ratio.json'
    generate_data(**kwargs, fib_len=fib_len, file_path=file_path)


    INPUT = "data/raman_simulator/3_pumps/100_fiber_0.0_ratio.json"
    OUTPUT = "data/raman_simulator/3_pumps/100_fiber_0.0_ratio_sorted.json"

    write_sorted_dataset_copy(INPUT, OUTPUT)
