import torch
import copy
import matplotlib

import raman_amplifier as ra
import custom_types as ct
from entry_points.train_models import get_or_train_forward_model

from ..controller_base import Controller


class GradientDescentController(Controller):
    def __init__(
        self,
        model_dir_path: str = "models/models/",
        training_data: str | None = None,
        lr_model: float = 1e-3,
        lr_control: float = 100,
        epochs: int = 200,
        batch_size: int = 32,
        iterations: int = 1000,
    ):
        super().__init__()
        self.control_lr = lr_control
        self.iterations = iterations
        assert isinstance(training_data, str)

        self.model = get_or_train_forward_model(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr_model,
            dir_path=model_dir_path
        )

    def get_control(
        self,
        curr_input: ra.RamanInputs,
        curr_output: ra.Spectrum[ct.Power],
        target_output: ra.Spectrum[ct.Power],
    ) -> ra.RamanInputs:
        
        x_np = copy.deepcopy(curr_input).normalize().as_array()
        x = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

        if not target_output.normalized:
            target_output.normalize()

        target_arr = target_output.as_array(include_freq=False)
        target = torch.tensor(target_arr, dtype=torch.float32).unsqueeze(0)

        self.loss_history: list[float] = []

        for _ in range(self.iterations):
            x = x.detach().requires_grad_(True)

            y_pred = self.model(x.unsqueeze(0))
            loss = torch.nn.functional.mse_loss(y_pred, target)

            self.loss_history.append(loss.item())

            loss.backward()

            with torch.no_grad():
                assert x.grad is not None
                x -= self.control_lr * x.grad
        
        x_final = x.detach().cpu().numpy()
        next_input = ra.RamanInputs.from_array(x_final).denormalize()

        control_delta = copy.deepcopy(next_input)
        control_delta -= curr_input

        if target_output.normalized:
            target_output.denormalize()

        return control_delta

    def update_controller(self, error: ra.Spectrum[ct.Power], control_delta: ra.RamanInputs) -> None:
        pass

    def plot_custom_data(self, ax: matplotlib.axes.Axes):
        # ---- Plot loss evolution ----
        ax.plot(self.loss_history)
        ax.set_xlabel("Optimization iteration")
        ax.set_ylabel("MSE loss")
        ax.set_title("Forward-model loss during control optimization")
        ax.grid(True)
