import os
from pathlib import Path
from typing import Optional

import models as m
from utils import parser

MODULE_NAME = "Model Trainer"



def _make_model_filename(models_path: str, dataset: str, epochs: int, learning_rate: float, *args, **kwargs):
    os.makedirs(models_path, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(dataset))[0]
    fname = f"forward_E{epochs}_lr{learning_rate}_dataset-{dataset_name}"
    return os.path.join(models_path, fname)



def find_latest_model(models_path: str, prefix: str, *args, **kwargs) -> Optional[str]:
    """
    Finds the latest model file matching prefix*.pt in models_path
    """
    models_path = Path(models_path)

    if not models_path.exists():
        return None

    candidates = list(models_path.glob(f"{prefix}*.pt"))
    if not candidates:
        return None

    # newest by modified time
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def get_or_train_forward_model(
    epochs: int = 200,
    batch_size: int = 32,
    model_dir: str = 'models/models',
    training_data_path: str = 'data/raman_simulator/3_pumps/100_fiber_0.0_ratio.json',
    prefix: str = "forward"
) -> m.ForwardNN:

    training_parser = parser.get_model_training_parser()
    args = training_parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if v is not None}

    model_dir = _make_model_filename(**kwargs, dataset=training_data_path)

    model_path = find_latest_model(prefix=prefix, **kwargs)

    model = m.ForwardNN(**kwargs)

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
    model.save(save_path)

    print(f"[{MODULE_NAME}] Forward model saved to {save_path}")
    return model


def get_or_train_backward_model(
    forward_model: m.ForwardNN,
    model_dir: str,
    lr: float,
    epochs: int,
    batch_size: int,
    training_data_path: str,
    prefix: str = "backward"
) -> m.BackwardNN:

    model_path = find_latest_model(model_dir, prefix)

    model = m.BackwardNN(forward_model=forward_model, lr=lr)

    if model_path is not None:
        print(f"[{MODULE_NAME}] Found backward model: {model_path}")
        model.load(model_path)
        return model

    print(f"[{MODULE_NAME}] No backward model found — training a new one...")

    final_loss = model.fit(
        training_data_path,
        epochs=epochs,
        batch_size=batch_size
    )

    save_path = os.path.join(model_dir, f"{prefix}_loss{final_loss:.6f}.pt")
    model.save(save_path)

    print(f"[{MODULE_NAME}] Backward model saved to {save_path}")
    return model
