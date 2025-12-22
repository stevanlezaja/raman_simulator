import matplotlib.pyplot as plt
import torch
import numpy as np

import raman_amplifier as ra
import models as m

from entry_points.train_models import get_or_train_backward_rpm_ensemble, get_or_train_backward_model
from utils.loading_data_from_file import load_raman_dataset

backward_model = get_or_train_backward_model()
backward_ensemble = m.BackwardEnsemble(get_or_train_backward_rpm_ensemble())

data_generator = load_raman_dataset('data/raman_simulator/3_pumps/100_fiber_0.0_ratio.json')
m.BackwardNN._prepare_dataset(backward_model, 'data/raman_simulator/3_pumps/100_fiber_0.0_ratio.json')

plt.ion()  # interactive mode

for step_idx, (target_raman_inputs, spectrum) in enumerate(data_generator):

    # Normalize spectrum once per sample
    spec_norm = spectrum.normalize()
    spec_array = spec_norm.as_array()
    spec_array = spec_array[len(spec_array)//2:]
    spec_tensor = torch.Tensor(spec_array)

    # Precompute normalized target for plotting
    target_norm_array = target_raman_inputs.normalize().as_array()

    # Backward NN prediction (once)
    bw_predicted_raman_inputs = ra.RamanInputs.from_array(
        backward_model.forward(spec_tensor).detach().numpy()
    ).denormalize()
    bw_norm_array = bw_predicted_raman_inputs.normalize().as_array()

    # Take 100 ensemble predictions per sample
    for i in range(100):
        for model in backward_ensemble.models:   # assuming `models` is a list of PyTorch modules
            model.train()
        bwe_predicted_raman_inputs = ra.RamanInputs.from_array(
            backward_ensemble.forward(spec_tensor).detach().numpy()
        ).denormalize()
        bwe_norm_array = bwe_predicted_raman_inputs.normalize().as_array()

        x = np.arange(6)
        width = 0.25

        plt.figure(figsize=(8, 4))
        plt.bar(x - width, target_norm_array, width, label="Target")
        plt.bar(x, bw_norm_array, width, label="Backward NN")
        plt.bar(x + width, bwe_norm_array, width, label="Backward Ensemble")

        plt.xticks(x, [f"v{i}" for i in range(6)])
        plt.ylabel("Value")
        plt.title(f"Raman Input Comparison - Sample {step_idx+1}, Step {i+1}")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()
        
        plt.waitforbuttonpress()  # wait until user closes or presses key/mouse
        plt.close()
