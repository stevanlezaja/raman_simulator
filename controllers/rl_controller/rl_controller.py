import copy
import itertools

import numpy as np
import torch

import custom_types as ct
import raman_amplifier as ra

from ..controller_base import Controller


class RLController(Controller):
    def __init__(self):
        super().__init__()
        self.sequence_length = 1000
        self.input_nodes = 6
        self.output_nodes = 6
        self.curr_reward = np.array([2, 0])
        self.step_size_dict: dict[str, ct.UnitProtocol] = {'power': ct.Power(0.2, 'mW'), 'wavelength': ct.Length(2, 'nm')}
        self.learning_rate = 1e-2
        self.weight_decay = 1e-1
        self.baseline = 1
        self.weights = np.zeros((self.input_nodes, self.output_nodes))
        self.step_size = self.build_step_vector(self.input_nodes // 2)

    def build_step_vector(self, n_pumps: int) -> np.ndarray:
        power_step = self.step_size_dict['power'].value
        wl_step = self.step_size_dict['wavelength'].value
        power_step = 0.001
        wl_step = 0.001

        return np.array(
            [power_step] * n_pumps + [wl_step] * n_pumps,
            dtype=float
        )

    def reward(
        self,
        curr_input: ra.RamanInputs,
        curr_output: ra.Spectrum[ct.Power],
        target_output: ra.Spectrum[ct.Power],
    ) -> float:

        def shape_difference(spec1: ra.Spectrum[ct.Power], spec2: ra.Spectrum[ct.Power]):
            int1 = ra.spectrum.integral(spec1).W
            scaled_spec1 = copy.deepcopy(spec1)

            int2 = ra.spectrum.integral(spec2).W
            scaled_spec2 = copy.deepcopy(spec2)

            difference_spec = scaled_spec1 / int1 - scaled_spec2 / int2

            return difference_spec.mean

        def integral_difference(spec1: ra.Spectrum[ct.Power], spec2: ra.Spectrum[ct.Power]):
            int_dif = ra.spectrum.integral(spec1).W - ra.spectrum.integral(spec2).W
            return int_dif if int_dif > 0 else - 10 * int_dif

        def wavelength_spread(wavelengths: list[ct.Length]):
            spread = 0
            for w1, w2 in itertools.combinations(wavelengths, 2):
                spread += abs(w1.nm - w2.nm) ** 0.5
            return spread

        # sh_dif = 1 * shape_difference(curr_output, target_output)

        # int_dif = integral_difference(curr_output, target_output)

        # wl_spread = 0 * wavelength_spread(curr_input.wavelengths)

        # loss = sh_dif + int_dif - wl_spread

        return -ra.mse(curr_output, target_output)

    def get_control(self, curr_input: ra.RamanInputs, curr_output: ra.Spectrum[ct.Power], target_output: ra.Spectrum[ct.Power]) -> ra.RamanInputs:
        weight_history = np.zeros((self.sequence_length, self.input_nodes, self.output_nodes)) 
        weight_history[0] = np.copy(self.weights)
        probabilities = np.zeros((self.sequence_length, self.output_nodes))
        new_output_state = np.zeros((self.sequence_length, self.output_nodes))
        
        input_state = np.zeros((self.sequence_length, self.input_nodes))

        history = np.zeros(self.sequence_length)
        history[0] = -np.inf
        reward_history = np.zeros(self.sequence_length)
        parameter_change = np.zeros((self.sequence_length, self.input_nodes, self.output_nodes))

        for t in range(1, self.sequence_length):
            prev_parameters = weight_history[t - 1]
            logits = prev_parameters.T @ input_state[t - 1]
            probabilities[t] = torch.sigmoid(torch.tensor(logits))
            new_output_state[t] = np.random.binomial(1, probabilities[t])
            input_state[t] = input_state[t - 1] + self.step_size * (2 * new_output_state[t] - 1)
            history[t] = self.reward(curr_input, curr_output, target_output)
            reward_history[t] = self.curr_reward[0] if history[t] > history[t - 1] else self.curr_reward[1]
            grad = np.outer(input_state[t - 1], (new_output_state[t] - probabilities[t]))
            parameter_change[t] = (reward_history[t] - self.baseline) * grad
            weight_history[t] = prev_parameters + self.learning_rate * parameter_change[t] - self.weight_decay * prev_parameters
        
        self.weights = np.copy(weight_history[-1])

        control = ra.RamanInputs.from_array(input_state[-1])

        control.denormalize_without_bias()

        return control

    def update_controller(self, error: ra.Spectrum[ct.Power], control_delta: ra.RamanInputs) -> None:
        pass
