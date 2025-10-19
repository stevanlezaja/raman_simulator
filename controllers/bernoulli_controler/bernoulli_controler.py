import torch

import raman_amplifier as ra
import custom_types as ct
import utils.parameter

from ..controller_base import Controller


class BernoulliController(torch.nn.Module, Controller):
    def __init__(self,
                 beta: int = 1,
                 weight_decay: float = 0.0,
                 input_dim: int = 2,
                 power_step: ct.Power = ct.Power(0.01, 'W'),
                 wavelength_step: ct.Length = ct.Length(5, 'nm'),
                 lr: float = 1e-1,
                 gamma: float = 0.9
                 ):
        self.power_step = power_step
        self.wavelength_step = wavelength_step
        self.input_dim = input_dim
        self.logits = torch.zeros(input_dim)
        self.learning_rate = lr
        self.gamma = gamma
        self.beta = beta
        self.best_reward = None
        self.weight_decay = weight_decay
        self.baseline = 0.0
        self.history = {'probs': [], 'rewards': []}
        self.avg_sample = torch.zeros_like(self.logits)
        self.prev_error: ra.Spectrum[ct.Power] | None = None
        self.output_integral: float = 0.0
        self.target_integral: float = 0.0

    def get_control(
        self,
        curr_input: ra.RamanInputs,
        curr_output:ra.Spectrum[ct.Power],
        target_output: ra.Spectrum[ct.Power],
    ) -> ra.RamanInputs:

        probs = torch.sigmoid(self.logits)
        self.history['probs'].append(probs.detach().numpy())

        self.target_integral = 0.0
        for power in target_output.values:
            self.target_integral += power.value

        self.output_integral = 0.0
        for power in curr_output.values:
            self.output_integral += power.value

        dist = torch.distributions.Bernoulli(probs)
        sample = dist.sample()
        power_sample = sample[:self.input_dim // 2]
        wavelength_sample = sample[self.input_dim // 2:]

        power_action = self.power_step.value * (power_sample * 2 - 1)
        wavelength_action = self.wavelength_step.value * (wavelength_sample * 2 - 1)

        powers_update: list[ct.Power] = []
        for power in power_action:
            powers_update.append(ct.Power(float(power), 'W'))

        wavelength_update: list[ct.Length] = []
        for wl in wavelength_action:
            wavelength_update.append(ct.Length(float(wl), 'm'))

        self.last_sample = sample.detach()

        return ra.RamanInputs(powers=powers_update, wavelengths=wavelength_update)

    def update_controller(
            self,
            error: ra.Spectrum[ct.Power],
            control_delta: ra.RamanInputs
        ) -> None:

        curr_loss = abs(error.mean)
        prev_loss = abs(self.prev_error.mean) if self.prev_error is not None else None

        if prev_loss is None:
            reward = 0.0
        else:
            reward = prev_loss - curr_loss

        self.baseline = self.gamma * self.baseline + (1 - self.gamma) * reward
        advantage = reward - self.baseline

        sample = getattr(self, "last_sample", None)
        if sample is None:
            self.prev_error = error
            return

        probs = torch.sigmoid(self.logits)
        eligibility = sample - probs

        update = self.learning_rate * advantage * eligibility - self.weight_decay * self.logits

        max_step = 0.2
        update = torch.clamp(update, min=-max_step, max=max_step)

        self.logits = self.logits + update

        self.prev_error = error

    def _populate_parameters(self) -> None:
        self.power_step = utils.parameter.get_unit_input(self.power_step, ct.Power(5, 'mW'), "Power Step")
        self.wavelength_step = utils.parameter.get_unit_input(self.wavelength_step, ct.Length(1, 'nm'), "Wavelength Step")

        self.learning_rate = utils.parameter.get_numeric_input(f"Input learning rate: \n(Default - {self.learning_rate})", self.learning_rate)
        self.beta = utils.parameter.get_numeric_input(f"Insert value for beta:", default=self.beta)
        self.gamma = utils.parameter.get_numeric_input(f"Insert value for gamma:", default=self.gamma)
        self.weight_decay = utils.parameter.get_numeric_input(f"Insert value for weight_decay:", default=self.weight_decay)

        return
