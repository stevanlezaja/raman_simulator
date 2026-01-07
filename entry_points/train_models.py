import os
from pathlib import Path
from typing import Optional, Any
import random
import numpy as np
import torch

import models as m
from utils import parser

MODULE_NAME = "Model Trainer"



def _make_model_filename(prefix: str, models_path: str, training_data_path: str, epochs: int, learning_rate: float, *args, **kwargs):
    os.makedirs(models_path, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(training_data_path))[0]
    fname = f"{prefix}_E{epochs}_lr{learning_rate}_dataset-{dataset_name}"
    return os.path.join(models_path, fname)



def find_latest_model(models_path: str, prefix: str, epochs: int, *args, **kwargs) -> Optional[str]:
    """
    Finds the latest model file matching prefix*.pt in models_path
    """
    models_path = Path(models_path)

    if not models_path.exists():
        return None

    candidates = list(models_path.glob(f"{prefix}*{epochs}*.pt"))
    if not candidates:
        return None

    # newest by modified time
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def get_or_train_forward_model(
    prefix: str = "forward",
    epochs: int = 500,
    batch_size: int = 32,
    lr: float = 1e-3,
    dir_path: str = "models/models/",
) -> m.ForwardNN:

    kwargs: dict[str, Any] = {
        'epochs': epochs,
        'learning_rate': lr,
        'batch_size': batch_size,
        'training_data_path': 'data/raman_simulator/3_pumps/100_fiber_0.0_ratio_sorted.json',
        'models_path': dir_path,
    }

    model_dir = _make_model_filename(prefix, **kwargs)

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

    model_dir = _make_model_filename(prefix, **kwargs)

    model_path = find_latest_model(prefix=prefix, **kwargs)

    model = m.BackwardNN(forward_model=forward_model, **kwargs)

    if model_path is not None:
        print(f"[{MODULE_NAME}] Found backward model: {model_path}")
        model.load(model_path)
        return model

    print(f"[{MODULE_NAME}] No backward model found — training a new one...")

    final_loss = model.fit(**kwargs)

    save_path = model_dir + f"_loss{final_loss:.6f}.pt"
    print(save_path)
    model.save(save_path)

    print(f"[{MODULE_NAME}] Backward model saved to {save_path}")
    return model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_or_train_backward_ensemble(
    ensemble_size: int = 10,
    prefix: str = "backward",
    base_seed: int = 42,
) -> list[m.BackwardNN]:
    """
    Trains or loads an ensemble of backward models.
    """

    forward_model = get_or_train_forward_model()

    training_parser = parser.get_model_training_parser()
    args = training_parser.parse_args()
    kwargs = {k: v for k, v in vars(args).items() if v is not None}

    models = []

    for i in range(ensemble_size):
        seed = base_seed + i
        set_seed(seed)

        ens_prefix = f"{prefix}_ens{i}"

        model_dir = _make_model_filename(ens_prefix, **kwargs)
        model_path = find_latest_model(prefix=ens_prefix, **kwargs)

        model = m.BackwardNN(
            forward_model=forward_model,
            **kwargs
        )

        if model_path is not None:
            print(f"[{MODULE_NAME}] Found backward model {i}: {model_path}")
            model.load(model_path)
            models.append(model)
            continue


        print(f"[{MODULE_NAME}] Training backward model {i} (seed={seed})...")

        final_loss = model.fit(**kwargs)

        save_path = model_dir + f"_loss{final_loss:.6f}.pt"
        model.save(save_path)

        print(f"[{MODULE_NAME}] Backward model {i} saved to {save_path}")

        models.append(model)

    print(models)
    return models

