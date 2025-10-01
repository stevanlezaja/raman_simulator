import torch

import raman_amplifier as ra
import custom_types as ct

from ..controller_base import Controller


class BernoulliController(torch.nn.Module, Controller):
    def __init__(self,
                 beta: int = 1,
                 weight_decay: float = 1e-5,
                 input_dim: int = 2,
                 power_step: ct.Power = ct.Power(1, 'mW'),
                 wavelength_step: ct.Length = ct.Length(0.1, 'nm'),
                 lr: float = 1e-3,
                 gamma: float = 0
                 ):
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
        # self.history = {'probs': [], 'rewards': []}
        self.avg_sample = torch.zeros_like(self.logits)
        self.prev_error: ra.Spectrum[ct.Power] | None = None
    
    def get_control(
        self,
        curr_input: ra.RamanInputs,
        curr_output:ra.Spectrum[ct.Power],
        target_output: ra.Spectrum[ct.Power],
    ) -> ra.RamanInputs:

        probs = torch.sigmoid(self.logits)
        # self.history['probs'].append(probs.detach().numpy())

        dist = torch.distributions.Bernoulli(probs)
        sample = dist.sample()
        power_sample = sample[:self.input_dim // 2]
        wavelength_sample = sample[self.input_dim // 2:]

        power_action = self.power_step.value * (power_sample * 2 - 1)
        wavelength_action = self.wavelength_step.value * (wavelength_sample * 2 - 1)
        print("WAVELENGHT STEP", self.wavelength_step)
        print("WAVELENGTH ACTION: ", wavelength_action)

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

        # Use the current error (the result after applying control_delta).
        # Compute scalar loss (MSE of spectrum). Convert to float for stability.
        curr_loss = float(error.mean ** 2)
        prev_loss = float(self.prev_error.mean ** 2) if self.prev_error is not None else None

        # Reward = improvement in loss (positive if loss went down).
        if prev_loss is None:
            # no previous to compare with — give zero reward (or small negative to encourage exploration)
            reward = 0.0
        else:
            reward = prev_loss - curr_loss
        
        print(reward)

        # Running baseline for variance reduction
        self.baseline = self.gamma * self.baseline + (1 - self.gamma) * reward
        advantage = reward - self.baseline  # zero-meaned advantage

        # Use the stored Bernoulli sample and current probs
        sample = getattr(self, "last_sample", None)
        if sample is None:
            # fallback: do nothing
            self.prev_error = error
            return

        probs = torch.sigmoid(self.logits)
        eligibility = sample - probs  # classic REINFORCE for Bernoulli

        # gradient step (weight decay applied as small L2 pull towards zero)
        update = self.learning_rate * advantage * eligibility - self.weight_decay * self.logits
        # optional gradient clipping on update magnitude:
        max_step = 0.2
        update = torch.clamp(update, min=-max_step, max=max_step)

        self.logits = self.logits + update

        # diagnostics (remove or change to logger)
        # print(f"prev_loss={prev_loss}, curr_loss={curr_loss}, reward={reward:.6f}, adv={advantage:.6f}")
        # print(f"probs={probs.detach().numpy()}, sample={sample.detach().numpy()}, logits={self.logits.detach().numpy()}")

        # store for next iteration
        self.prev_error = error
