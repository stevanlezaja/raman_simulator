import os
from pathlib import Path
from typing import Optional

import models as m
from utils import parser

MODULE_NAME = "Model Trainer"



def _make_model_filename(models_path: str, training_data_path: str, epochs: int, learning_rate: float, *args, **kwargs):
    os.makedirs(models_path, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(training_data_path))[0]
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
    prefix: str = "forward"
) -> m.ForwardNN:

    training_parser = parser.get_model_training_parser()
    args = training_parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if v is not None}

    model_dir = _make_model_filename(**kwargs)

    model_path = find_latest_model(prefix=prefix, **kwargs)

    model = m.ForwardNN(**kwargs)

    if model_path is not None:
        print(f"[{MODULE_NAME}] Found forward model: {model_path}")
        model.load(model_path)
        return model

    print(f"[{MODULE_NAME}] No forward model found — training a new one...")

    final_loss = model.fit(**kwargs)

    save_path = model_dir + f"_loss{final_loss:.6f}.pt"
    model.save(save_path)

    print(f"[{MODULE_NAME}] Forward model saved to {save_path}")
    return model


def get_or_train_backward_model(
    prefix: str = "backward"
) -> m.BackwardNN:

    forward_model = get_or_train_forward_model()

    training_parser = parser.get_model_training_parser()
    args = training_parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if v is not None}

    model_dir = _make_model_filename(**kwargs)

    model_path = find_latest_model(prefix=prefix, **kwargs)

    model = m.BackwardNN(forward_model=forward_model, **kwargs)

    if model_path is not None:
        print(f"[{MODULE_NAME}] Found backward model: {model_path}")
        model.load(model_path)
        return model

    print(f"[{MODULE_NAME}] No backward model found — training a new one...")

    final_loss = model.fit(**kwargs)

    save_path = os.path.join(model_dir, f"{prefix}_loss{final_loss:.6f}.pt")
    model.save(save_path)

    print(f"[{MODULE_NAME}] Backward model saved to {save_path}")
    return model
