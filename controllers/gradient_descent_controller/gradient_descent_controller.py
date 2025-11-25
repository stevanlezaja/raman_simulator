import torch

import raman_amplifier as ra
import custom_types as ct
from utils.loading_data_from_file import load_raman_dataset

from ..controller_base import Controller
from .forward_nn import ForwardNN


class GradientDescentController(Controller):
    def __init__(self, lr_model: float = 1e-3, lr_control: float = 5e-2):
        super().__init__()
        self.model = ForwardNN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_model)
        self.control_lr = lr_control

    def train_controller(self, file_path: str, epochs: int = 100, batch_size: int = 32):
        """
        Train ForwardNN to approximate the real Raman simulator.
        Dataset entries: (RamanInputs, Spectrum[Power])
        """

        samples = list(load_raman_dataset(file_path))

        X_list = []
        Y_list = []

        for raman_inputs, spectrum in samples:
            x = torch.tensor(raman_inputs.normalize().as_array(), dtype=torch.float32)
            arr = spectrum.as_array()
            values = arr[len(arr)//2:]
            y = torch.tensor(values, dtype=torch.float32)

            X_list.append(x)
            Y_list.append(y)

        X = torch.stack(X_list)
        Y = torch.stack(Y_list)

        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_fn = torch.nn.MSELoss()

        for epoch in range(epochs):
            total = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                self.optimizer.step()
                total += loss.item()

            print(f"[GDController] epoch {epoch+1}/{epochs}, loss={total/len(loader):.6f}")


    def get_control(
        self,
        curr_input: ra.RamanInputs,
        curr_output: ra.Spectrum[ct.Power],
        target_output: ra.Spectrum[ct.Power]
    ) -> ra.RamanInputs:

        x = torch.tensor(curr_input.normalize().as_array(), dtype=torch.float32, requires_grad=True)

        arr = target_output.as_array()
        values = arr[len(arr)//2:]
        target = torch.tensor(values, dtype=torch.float32)
        y_pred = self.model(x)

        loss = torch.nn.functional.mse_loss(y_pred, target)
        loss.backward()
        grad_x = x.grad

        with torch.no_grad():
            x_new = x - self.control_lr * grad_x

        control = ra.RamanInputs.from_array(x_new.detach().numpy()).denormalize()
        print("CONTROL \n", control)
        return control

    def update_controller(self, error: ra.Spectrum[ct.Power], control_delta: ra.RamanInputs) -> None:
        pass