import os
import torch
import copy

import raman_amplifier as ra
import custom_types as ct

from ..controller_base import Controller
from .forward_nn import ForwardNN


class GradientDescentController(Controller):
    def __init__(
        self,
        model_path: str = "controllers/gradient_descent_controller/models/",
        training_data: str | None = None,
        lr_model: float = 1e-3,
        lr_control: float = 100,
        epochs: int = 200,
        batch_size: int = 32,
    ):
        super().__init__()
        self.model = ForwardNN(lr=lr_model)
        self.control_lr = lr_control

        # If the path is a directory, check if it contains any .pt model
        if os.path.isdir(model_path):
            existing = [f for f in os.listdir(model_path) if f.endswith(".pt")]
            if existing:
                latest = sorted(existing)[-1]   # or pick the best, or newest
                full_path = os.path.join(model_path, latest)

                print(f"[GDController] Found existing model — loading {latest}")
                self.model.load(full_path)
                return

        # No existing model found → need to train
        if training_data is None:
            raise ValueError("No model found and no training data provided.")

        print("[GDController] No model found — training a new one...")

        final_loss = self.model.fit(training_data, epochs=epochs, batch_size=batch_size)
        save_path = self._make_model_filename(model_path, training_data, epochs, final_loss)

        self.model.save(save_path)
        print(f"[GDController] Model saved as: {save_path}")

    def _make_model_filename(self, base_dir: str, dataset: str, epochs: int, loss: float):
        os.makedirs(base_dir, exist_ok=True)
        dataset_name = os.path.splitext(os.path.basename(dataset))[0]
        fname = f"forward_E{epochs}_L{loss:.4f}_dataset-{dataset_name}.pt"
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
