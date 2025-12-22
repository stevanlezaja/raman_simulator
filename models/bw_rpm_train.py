import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import raman_amplifier as ra
from models.forward_nn import ForwardNN
from utils.loading_data_from_file import load_raman_dataset

# -----------------------------
# Backward RPM Model definition
# -----------------------------
class BackwardRPMNN(nn.Module):
    def __init__(self, input_size=20, output_size=6, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Ensemble wrapper
# -----------------------------
class BackwardRPMEnsemble:
    def __init__(self, n_models=10, input_size=40, output_size=6, hidden_size=128):
        self.models = [BackwardRPMNN(input_size, output_size, hidden_size) for _ in range(n_models)]

    def train_ensemble(self, dataloader, epochs=100, lr=1e-3, device='cpu'):
        self.device = device
        for model in self.models:
            model.to(device)

        optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in self.models]
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for spec_tensor, target_tensor in dataloader:
                spec_tensor = spec_tensor.to(device)
                target_tensor = target_tensor.to(device)

                for model, optimizer in zip(self.models, optimizers):
                    optimizer.zero_grad()
                    pred = model(spec_tensor)
                    loss = loss_fn(pred, target_tensor)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/len(dataloader):.6f}")

    def predict(self, x):
        # average predictions over ensemble
        with torch.no_grad():
            preds = [m(x) for m in self.models]
            return torch.mean(torch.stack(preds), dim=0)

    def save(self, folder='models/ensemble'):
        os.makedirs(folder, exist_ok=True)
        for i, m in enumerate(self.models):
            torch.save(m.state_dict(), os.path.join(folder, f'model_{i}.pt'))

    def load(self, folder='models/ensemble'):
        for i, m in enumerate(self.models):
            m.load_state_dict(torch.load(os.path.join(folder, f'model_{i}.pt')))

# -----------------------------
# Prepare dataset
# -----------------------------
def prepare_dataset(json_path='data/raman_simulator/3_pumps/100_fiber_0.0_ratio.json'):
    data_generator = load_raman_dataset(json_path)
    ForwardNN._prepare_dataset(ForwardNN(), 'data/raman_simulator/3_pumps/100_fiber_0.0_ratio.json')

    X_list, Y_list = [], []
    for raman_inputs, spectrum in data_generator:
        spec_array = spectrum.normalize().as_array()
        spec_array = spec_array[len(spec_array)//2:]  # take last half
        X_list.append(spec_array)
        Y_list.append(raman_inputs.normalize().as_array())

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    Y = torch.tensor(np.stack(Y_list), dtype=torch.float32)

    dataset = TensorDataset(X, Y)
    return dataset

# -----------------------------
# Main training routine
# -----------------------------
if __name__ == "__main__":
    dataset = prepare_dataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    ensemble = BackwardRPMEnsemble(n_models=10)
    ensemble.train_ensemble(train_loader, epochs=50, lr=1e-3)

    ensemble.save('models/backward_rpm_ensemble')

    # Example prediction
    x_example, y_example = next(iter(val_loader))
    pred = ensemble.predict(x_example)
    print("Predicted:", pred[0].numpy())
    print("Target   :", y_example[0].numpy())
