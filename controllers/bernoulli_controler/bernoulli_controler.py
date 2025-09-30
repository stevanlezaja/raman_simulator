import torch

import raman_amplifier as ra
import custom_types as ct

from ..controller_base import Controller


class BernoulliController(torch.nn.Module, Controller):
    def __init__(self,
                 model: ra.RamanAmplifier,
                 beta: int = 1,
                 weight_decay: float = 1e-5,
                 input_dim: int = 2,
                 power_step: float = 0.1,
                 wavelength_step: float = 0.1,
                 lr: float = 1e-2,
                 gamma: float = 0
                 ):
        self.model = model
        self.power_step = power_step
        self.wavelength_step = wavelength_step
        self.input_dim = input_dim
        self.logits = 0.2 * torch.randn(input_dim)
        self.learning_rate = lr
        self.gamma = gamma
        self.beta = beta
        self.best_reward = None
        self.weight_decay = weight_decay
        self.baseline = 0.0
        self.history = {'probs': [], 'rewards': []}
        self.avg_sample = torch.zeros_like(self.logits)
        self.prev_error = None
    
    def get_control(
        self,
        curr_input: ra.RamanInputs,
        curr_output:ra.Spectrum[ct.Power],
        target_output: ra.Spectrum[ct.Power],
    ) -> ra.RamanInputs:

        probs = torch.sigmoid(self.logits)
        self.history['probs'].append(probs.detach().numpy())

        dist = torch.distributions.Bernoulli(probs)
        sample = dist.sample()
        power_sample = sample[:self.input_dim // 2]
        wavelength_sample = sample[self.input_dim // 2:]

        power_action = self.power_step * (power_sample * 2 - 1)
        wavelength_action = self.power_step * (wavelength_sample * 2 - 1)
        return ra.RamanInputs(powers=power_action, wavelengths=wavelength_action)

    def update_controller(
            self,
            error: GainSpectrum,
            x_delta: RaInputs
    ):
        sample = x_delta.value / self.step / 2 + 1
        loss = -1 * torch.norm(self.prev_error.value)**2 if self.prev_error is not None else 0.0
        reward = -loss

        reinforcement_factor = reward - self.baseline - self.beta
        self.baseline = self.gamma * self.baseline + (1 - self.gamma) * reward

        eligibility = sample - self.avg_sample
        self.avg_sample = self.gamma * self.avg_sample + (1 - self.gamma) * sample
        self.logits += torch.clamp(self.learning_rate * reinforcement_factor * eligibility - self.weight_decay * self.logits, min=-0.1, max=0.1)
        self.prev_error = error