import torch

import raman_amplifier as ra
import custom_types as ct

from ..controller_base import Controller


class BernoulliController(torch.nn.Module, Controller):
    def __init__(self,
                 power_step: ct.Power = ct.Power(1, 'mW'),
                 wavelength_step: ct.Length = ct.Length(5, 'nm'),
                 lr: float = 1e-1,
                 weight_decay: float = 0.0,
                 beta: int = 1,
                 gamma: float = 1,
                 input_dim: int = 2,
                 ):
        Controller.__init__(self)
        self._params['power_step'] = (ct.Power, power_step)
        self._params['wavelength_step'] = (ct.Length, wavelength_step)
        self._params['lr'] = (float, lr)
        self._params['weight_decay'] = (float, weight_decay)
        self._params['beta'] = (float, beta)
        self._params['gamma'] = (float, gamma)

        self.input_dim = input_dim
        self.logits = torch.zeros(input_dim)
        self.best_reward = None
        self.baseline = 0.0
        self.history = {'probs': [], 'rewards': []}
        self.avg_sample = torch.zeros_like(self.logits)
        self.prev_error: ra.Spectrum[ct.Power] | None = None
        self.output_integral: float = 0.0
        self.target_integral: float = 0.0

    @property
    def power_step(self) -> ct.Power:
        return self._params['power_step'][1]

    @property
    def wavelength_step(self) -> ct.Length:
        return self._params['wavelength_step'][1]

    @property
    def learning_rate(self) -> float:
        return self._params['lr'][1]

    @property
    def weight_decay(self) -> float:
        return self._params['weight_decay'][1]

    @property
    def beta(self) -> float:
        return self._params['beta'][1]

    @property
    def gamma(self) -> float:
        return self._params['gamma'][1]

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
