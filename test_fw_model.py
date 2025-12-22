import torch
import matplotlib.pyplot as plt

import custom_types as ct
import raman_amplifier as ra
import models as m

from entry_points.train_models import get_or_train_forward_model
from utils.loading_data_from_file import load_raman_dataset


forward_model = get_or_train_forward_model()

data_generator = load_raman_dataset('data/raman_simulator/3_pumps/100_fiber_0.0_ratio.json')
m.ForwardNN._prepare_dataset(forward_model, 'data/raman_simulator/3_pumps/100_fiber_0.0_ratio.json')

for raman_inputs, target_spectrum in data_generator:
    ra_norm = raman_inputs.normalize()
    ra_arr = ra_norm.as_array()
    ra_tens = torch.Tensor(ra_arr)

    fw_model_output = list(forward_model.forward(ra_tens).detach().numpy())
    power_pred = [ct.Power(p, 'W') for p in fw_model_output]
    spec_pred = ra.Spectrum(value_cls=ct.Power, frequencies=target_spectrum.frequencies, values=power_pred).denormalize()

    fig = plt.figure()

    plt.plot( # type: ignore
        [f.Hz for f in target_spectrum.frequencies],
        [val.mW for val in target_spectrum.values],
        label="Target",
    )
    plt.plot( # type: ignore
        [f.Hz for f in spec_pred.frequencies],
        [val.mW for val in spec_pred.values],
        label="Current Output",
    )
    plt.grid()  # type: ignore
    plt.legend()  # type: ignore
    plt.show()
