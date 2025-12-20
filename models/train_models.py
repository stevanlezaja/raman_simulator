import os
from pathlib import Path
from typing import Optional

from .forward_nn import ForwardNN
from .backward_nn import BackwardNN


MODULE_NAME = "Model Trainer"


def find_latest_model(model_dir: str, prefix: str) -> Optional[str]:
    """
    Finds the latest model file matching prefix*.pt in model_dir
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        return None

    candidates = list(model_dir.glob(f"{prefix}*.pt"))
    if not candidates:
        return None

    # newest by modified time
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def get_or_train_forward_model(
    model_dir: str,
    lr: float,
    epochs: int,
    batch_size: int,
    training_data_path: str,
    prefix: str = "forward"
) -> ForwardNN:

    model_path = find_latest_model(model_dir, prefix)

    model = ForwardNN(lr=lr)

    if model_path is not None:
        print(f"[{MODULE_NAME}] Found forward model: {model_path}")
        model.load(model_path)
        return model

    print(f"[{MODULE_NAME}] No forward model found — training a new one...")

    final_loss = model.fit(
        training_data_path,
        epochs=epochs,
        batch_size=batch_size
    )

    save_path = model_dir + f"_loss{final_loss:.6f}.pt"
    print("\n\n\n", save_path, "\n\n\n")
    model.save(save_path)

    print(f"[{MODULE_NAME}] Forward model saved to {save_path}")
    return model


def train_backward_model(fw_model: ForwardNN, lr: float, epochs: int, batch_size: int, training_data_path: str | None, save_model_path: str):
    model = BackwardNN(forward_model=fw_model, lr=lr)

    if training_data_path is None:
        raise ValueError("No model found and no training data provided.")

    print(f"[{MODULE_NAME}] No model found — training a new one...")

    final_loss = model.fit(training_data_path, epochs=epochs, batch_size=batch_size)

    model.save(f"{save_model_path}_loss{final_loss}.pt")
    print(f"[{MODULE_NAME}] Model saved as: {save_model_path}, with the loss of {final_loss}")
