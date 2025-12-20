import os
import torch
import copy

import raman_amplifier as ra
import custom_types as ct
import models as m

from ..controller_base import Controller


class GradientDescentController(Controller):
    def __init__(
        self,
        model_path: str = "models/models/",
        training_data: str | None = None,
        lr_model: float = 1e-3,
        lr_control: float = 100,
        epochs: int = 200,
        batch_size: int = 32,
    ):
        super().__init__()
        self.control_lr = lr_control
        assert isinstance(training_data, str)
        save_path = self._make_model_filename(model_path, training_data, epochs)

        self.model = m.get_or_train_forward_model(save_path, lr_model, epochs, batch_size, training_data)

    def _make_model_filename(self, base_dir: str, dataset: str, epochs: int):
        os.makedirs(base_dir, exist_ok=True)
        dataset_name = os.path.splitext(os.path.basename(dataset))[0]
        fname = f"forward_E{epochs}_L_dataset-{dataset_name}"
        return os.path.join(base_dir, fname)

    def get_control(
        self,
        curr_input: ra.RamanInputs,
        curr_output: ra.Spectrum[ct.Power],
        target_output: ra.Spectrum[ct.Power]
    ) -> ra.RamanInputs:
        x0 = copy.deepcopy(curr_input).normalize().as_array()
        x_leaf = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
        x = x_leaf.unsqueeze(0)

        target_arr = target_output.as_array()[-40:]       # SAFETY: ensure this matches model output dim
        target = torch.tensor(target_arr, dtype=torch.float32).unsqueeze(0)

        y_pred = self.model(x)
        loss = torch.nn.functional.mse_loss(y_pred, target)
        loss.backward()

        grad = x_leaf.grad

        with torch.no_grad():
            x_delta = - self.control_lr * grad

        x_new_np = x_leaf.detach() + x_delta.detach()

        next_input = ra.RamanInputs.from_array(x_new_np).denormalize()

        control_delta = copy.deepcopy(next_input)
        control_delta -= curr_input

        return control_delta

    def update_controller(self, error: ra.Spectrum[ct.Power], control_delta: ra.RamanInputs) -> None:
        pass
