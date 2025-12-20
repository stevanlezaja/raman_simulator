from .forward_nn import ForwardNN
from .backward_nn import BackwardNN


def train_forward_model(lr: float, epochs: int, batch_size: int, training_data_path: str | None, save_model_path: str):
    model = ForwardNN(lr=lr)

    if training_data_path is None:
        raise ValueError("No model found and no training data provided.")

    print("[GDController] No model found — training a new one...")

    final_loss = model.fit(training_data_path, epochs=epochs, batch_size=batch_size)

    model.save(f"{save_model_path}_loss{final_loss}.pt")
    print(f"[GDController] Model saved as: {save_model_path}, with the loss of {final_loss}")


def train_backward_model(fw_model: ForwardNN, lr: float, epochs: int, batch_size: int, training_data_path: str | None, save_model_path: str):
    model = BackwardNN(forward_model=fw_model, lr=lr)

    if training_data_path is None:
        raise ValueError("No model found and no training data provided.")

    print("[GDController] No model found — training a new one...")

    final_loss = model.fit(training_data_path, epochs=epochs, batch_size=batch_size)

    model.save(f"{save_model_path}_loss{final_loss}.pt")
    print(f"[GDController] Model saved as: {save_model_path}, with the loss of {final_loss}")
