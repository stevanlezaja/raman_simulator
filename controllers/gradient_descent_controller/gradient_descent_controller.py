import os
import torch
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
        lr_control: float = 1e-10,
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

        x = torch.tensor(
            curr_input.normalize().as_array(),
            dtype=torch.float32,
            requires_grad=True,
        )

        arr = target_output.as_array()
        target = torch.tensor(arr[len(arr)//2:], dtype=torch.float32)

        y_pred = self.model(x)
        loss = torch.nn.functional.mse_loss(y_pred, target)
        loss.backward()

        with torch.no_grad():
            x_new = x - self.control_lr * x.grad

        control = ra.RamanInputs.from_array(x_new.detach().numpy()).denormalize()
        return control

    def update_controller(self, error: ra.Spectrum[ct.Power], control_delta: ra.RamanInputs) -> None:
        pass