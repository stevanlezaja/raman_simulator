"""
bernoulli_controller.py
=======================

Implements a Bernoulli-based reinforcement learning controller for Raman
Amplifiers. The controller generates stochastic control signals for pump
powers and wavelengths based on a Bernoulli policy and updates its internal
parameters using policy gradient-like updates.

The module depends on `torch` for tensor operations and probabilistic sampling,
`custom_types` for units like Power and Length, and the abstract `Controller`
base class.

Example:
    >>> from controllers.bernoulli_controller import BernoulliController
    >>> controller = BernoulliController()
    >>> controller.populate_parameters()
"""

import copy
import torch
import matplotlib.axes
import numpy as np
import itertools

import raman_amplifier as ra
import custom_types as ct

from ..controller_base import Controller


class BernoulliController(torch.nn.Module, Controller):
    """
    A Bernoulli stochastic policy controller for Raman Amplifiers.

    This controller represents pump power and wavelength adjustments as
    Bernoulli random variables derived from logits. It supports interactive
    parameter population via the `_params` dictionary, and updates its internal
    policy using a simple reward baseline and gradient-like update.

    Parameters
    ----------
    power_step : ct.Power, optional
        The step size for power adjustments (default 1 mW).
    wavelength_step : ct.Length, optional
        The step size for wavelength adjustments (default 5 nm).
    lr : float, optional
        Learning rate for policy updates (default 0.1).
    weight_decay : float, optional
        L2 regularization for logits (default 0.0).
    beta : float, optional
        Coefficient for optional reward scaling (default 1.0).
    gamma : float, optional
        Reward baseline decay factor for advantage estimation (default 1.0).
    input_dim : int, optional
        Dimension of the input vector for the Bernoulli distribution
        (default 2).

    Attributes
    ----------
    _params : dict[str, tuple[type, Any]]
        Stores controller parameters with their types and values.
    logits : torch.Tensor
        The raw logits used to sample Bernoulli actions.
    history : dict
        Stores historical probabilities and rewards for analysis.
    baseline : float
        Running reward baseline for advantage estimation.
    prev_error : ra.Spectrum[ct.Power] | None
        Stores the previous error spectrum for reward calculation.
    """

    def __init__(self, *,
                 power_step: ct.Power = ct.Power(1, 'mW'),
                 wavelength_step: ct.Length = ct.Length(5, 'nm'),
                 lr: float = 1e-1,
                 weight_decay: float = 0.0,
                 beta: float = 1,
                 gamma: float = 1,
                 input_dim: int = 2,
                 ):
        super(torch.nn.Module, self).__init__()
        Controller.__init__(self)
        self._params['power_step'] = (ct.Power, power_step)
        self._params['wavelength_step'] = (ct.Length, wavelength_step)
        self._params['lr'] = (float, lr)
        self._params['weight_decay'] = (float, weight_decay)
        self._params['beta'] = (float, beta)
        self._params['gamma'] = (float, gamma)

        self.input_dim = input_dim
        self.logits = 0.01 * torch.randn(input_dim)
        self.best_reward = None
        self._baseline = 0.0
        self.history: dict[str, list[float]|dict[str, list[float]]] = {'probs': [], 'rewards': {'total': [], 'shape_loss': [], 'integral_loss': [], 'mse_loss': [], 'wavelength_spread': []}, 'baseline': []}
        self.avg_sample = torch.zeros_like(self.logits)
        self.prev_error: ra.Spectrum[ct.Power] | None = None
        self.output_integral: float = 0.0
        self.target_integral: float = 0.0
        self.last_sample = None

    @property
    def power_step(self) -> ct.Power:
        """
        Get the current power step parameter.

        Returns
        -------
        ct.Power
            The step size for power adjustments used by the controller.
        """
        return self._params['power_step'][1]

    @property
    def wavelength_step(self) -> ct.Length:
        """
        Get the current wavelength step parameter.

        Returns
        -------
        ct.Length
            The step size for wavelength adjustments used by the controller.
        """
        return self._params['wavelength_step'][1]

    @property
    def learning_rate(self) -> float:
        """
        Get the learning rate for policy updates.

        Returns
        -------
        float
            The learning rate applied to Bernoulli logits during controller updates.
        """
        return self._params['lr'][1]

    @property
    def weight_decay(self) -> float:
        """
        Get the L2 regularization coefficient for logits.

        Returns
        -------
        float
            The weight decay applied to logits during policy updates.
        """
        return self._params['weight_decay'][1]

    @property
    def beta(self) -> float:
        """
        Get the beta parameter for reward scaling.

        Returns
        -------
        float
            The reward scaling coefficient used internally by the controller.
        """
        return self._params['beta'][1]

    @property
    def gamma(self) -> float:
        """
        Get the gamma parameter for reward baseline decay.

        Returns
        -------
        float
            The discount factor applied to the running baseline of rewards.
        """
        return self._params['gamma'][1]

    @property
    def rewards(self) -> list[float]:
        return self.history['rewards']['total']

    @property
    def baseline(self) -> list[float]:
        return self.history['baseline']

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
                spread += abs(w1.nm - w2.nm) **0.5
            return spread

        sh_dif = 1 * shape_difference(curr_output, target_output)

        int_dif = integral_difference(curr_output, target_output)

        wl_spread = 0 * wavelength_spread(curr_input.wavelengths)

        self.history['rewards']['shape_loss'].append(sh_dif)
        self.history['rewards']['integral_loss'].append(int_dif)
        self.history['rewards']['wavelength_spread'].append(wl_spread)

        loss = sh_dif + int_dif - wl_spread

        # print(f"Reward is: {-loss}\n  Shape difference is {sh_dif/loss*100:.2f}%\n  Integral difference is {int_dif/loss*100:.2f}%\n")

        return -loss

    def get_control(
        self,
        curr_input: ra.RamanInputs,
        curr_output: ra.Spectrum[ct.Power],
        target_output: ra.Spectrum[ct.Power],
    ) -> ra.RamanInputs:
        """
        Generate a new RamanInputs control signal based on a Bernoulli policy.

        The controller computes probabilities via the sigmoid of the internal logits,
        samples a Bernoulli action, and converts the resulting +/-1 actions into
        stepwise changes in power and wavelength.

        Parameters
        ----------
        curr_input : ra.RamanInputs
            The last applied control signal.
        curr_output : ra.Spectrum[ct.Power]
            The last measured gain spectrum.
        target_output : ra.Spectrum[ct.Power]
            The desired gain spectrum to achieve.

        Returns
        -------
        ra.RamanInputs
            The new control signal for pump powers and wavelengths.
        """
        self.curr_input = curr_input
        self.curr_output = curr_output
        self.target_output = target_output

        probs = torch.sigmoid(self.logits)
        self.history['probs'].append(probs.detach().numpy()) # type: ignore

        dist = torch.distributions.Bernoulli(probs)

        sample = torch.zeros_like(dist.sample())
        num_samples = 50
        for _ in range(num_samples):
            sample += dist.sample()
        sample /= num_samples

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
        """
        Update the controller's internal policy using the observed error.

        This method computes a reward based on the reduction in absolute error
        since the previous step, updates a running baseline using the gamma
        decay factor, computes the advantage, and applies a policy gradient-like
        update to the logits. Updates are clamped to avoid excessive step sizes.

        Parameters
        ----------
        error : ra.Spectrum[ct.Power]
            The difference between the target and measured gain spectrum.
        control_delta : ra.RamanInputs
            The previously applied change in control inputs.

        Notes
        -----
        The controller stores the last error and logits for use in subsequent
        updates. The method implements a simple stochastic policy gradient
        update with optional weight decay and step clamping.
        """

        reward = self.reward(self.curr_input, self.curr_output, self.target_output)
        self.history['rewards']['total'].append(reward)

        mse = ra.spectrum.mse(self.curr_output, self.target_output)
        self.history['rewards']['mse_loss'].append(mse)

        self._baseline = self.gamma * self._baseline + (1 - self.gamma) * reward
        self.history['baseline'].append(self._baseline)

        advantage = self.beta * (reward - self._baseline)

        self.prev_error = error

        sample = getattr(self, "last_sample", None)
        if sample is None:
            return

        probs = torch.sigmoid(self.logits)
        eligibility = sample - probs

        update = self.learning_rate * advantage * eligibility - self.weight_decay * self.logits

        self.logits += update

    def plot_loss(self, ax: matplotlib.axes.Axes) -> None:
        ax.plot(self.rewards, label='Reward')  # type: ignore
        ax.plot(self.baseline, label='Baseline')  # type: ignore
        # ax.plot(self.history['rewards']['mse_loss'], label='MSE Loss')  # type: ignore
        ax.plot([-x for x in self.history['rewards']['integral_loss']], label='Integral Loss')  # type: ignore
        ax.plot([-x for x in self.history['rewards']['shape_loss']], label='Shape Loss')  # type: ignore
        ax.set_xlabel("Iteration")  # type: ignore
        ax.set_ylabel("Reward")  # type: ignore
        ax.set_title("Reward over time")  # type: ignore
        ax.grid()  # type: ignore
        ax.legend()  # type: ignore

    def plot_custom_data(self, ax: matplotlib.axes.Axes):
        probs = np.array(self.history['probs'])  # shape: (steps, n_actions)
        # --- Step probability evolution ---
        ax.plot(probs[:, :])  # type: ignore
        ax.set_xlabel("Iteration")  # type: ignore
        ax.set_ylabel("Probability")  # type: ignore
        ax.set_title("Step probability evolution")  # type: ignore
        ax.grid()  # type: ignore
        ax.legend()  # type: ignore

    def converged(self, thresholds: tuple[float, float], num_steps: int, min_steps: int) -> bool:
        converged = False
        assert min_steps > num_steps
        if len(self.history['probs']) > min_steps:
            converged = True
            for x in self.history['probs'][::-num_steps]:
                for y in x:
                    if not (thresholds[0] < y < thresholds[1]):
                        converged = False
                        break
        if converged:
            print(self.history['probs'][:num_steps])
            print(self.history['probs'])
        return converged
