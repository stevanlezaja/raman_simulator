import torch

import raman_amplifier as ra
import models as m

from entry_points.train_models import get_or_train_backward_ensemble, get_or_train_backward_model
from utils.loading_data_from_file import load_raman_dataset


backward_model = get_or_train_backward_model()
backward_ensemble = m.BackwardEnsemble(get_or_train_backward_ensemble())

data_generator = load_raman_dataset('data/raman_simulator/3_pumps/100_fiber_0.0_ratio.json')
m.BackwardNN._prepare_dataset(backward_model, 'data/raman_simulator/3_pumps/100_fiber_0.0_ratio.json')

for target_raman_inputs, spectrum in data_generator:
    spec_norm = spectrum.normalize()
    spec_array = spec_norm.as_array()
    spec_array = spec_array[len(spec_array)//2:]
    spec_tensor = torch.Tensor(spec_array)

    print(target_raman_inputs)
    bw_predicted_raman_inputs = ra.RamanInputs.from_array(backward_model.forward(spec_tensor).detach().numpy()).denormalize()
    print(bw_predicted_raman_inputs)
    bwe_predicted_raman_inputs = ra.RamanInputs.from_array(backward_ensemble.forward(spec_tensor).detach().numpy()).denormalize()
    print(bwe_predicted_raman_inputs)
    _ = input()

