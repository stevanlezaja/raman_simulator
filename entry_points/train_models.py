import os
from pathlib import Path
from typing import Optional
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
    prefix: str = "forward"
) -> m.ForwardNN:

    training_parser = parser.get_model_training_parser()
    args = training_parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if v is not None}

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


from typing import List
from pathlib import Path
from models import ForwardNN, BackwardRPM


MODEL_DIR = Path("models/backward_rpm_ensemble")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

def get_or_train_backward_rpm_ensemble(
    ensemble_size: int = 10,
    steps: int = 200,
    lr: float = 5e-3,
    proj_dim: int = 8,
    clamp: bool = True,
    force_retrain: bool = False,
) -> List[BackwardRPM]:
    """
    Returns a list of BackwardRPM instances (ensemble).
    Each instance has a different random projection matrix R.
    """
    forward_model = get_or_train_forward_model()

    ensemble_models = []

    for i in range(ensemble_size):
        model_path = MODEL_DIR / f"backward_rpm_{i}.pt"

        if model_path.exists() and not force_retrain:
            # load existing
            model = BackwardRPM(
                forward_model=forward_model,
                steps=steps,
                lr=lr,
                proj_dim=proj_dim,
                clamp=clamp,
            )
            model.load_state_dict(torch.load(model_path))
        else:
            # create new RPM
            model = BackwardRPM(
                forward_model=forward_model,
                steps=steps,
                lr=lr,
                proj_dim=proj_dim,
                clamp=clamp,
            )
            # save it
            torch.save(model.state_dict(), model_path)

        ensemble_models.append(model)

    return ensemble_models


# Example usage:
if __name__ == "__main__":
    forward_model = ForwardNN()
    # load your trained forward model here
    forward_model.load_state_dict(torch.load("models/forward_model.pt"))

    ensemble = get_or_train_backward_rpm_ensemble(
        forward_model,
        ensemble_size=5,
        steps=300,
        lr=5e-3,
        proj_dim=8,
    )

    # predict with the first ensemble member
    spectrum = torch.rand(40)  # example normalized spectrum
    u_pred = ensemble[0].forward(spectrum)
    print(u_pred)
