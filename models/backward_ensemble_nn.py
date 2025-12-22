import torch

from .backward_nn import BackwardNN

class BackwardEnsemble:
    def __init__(self, models: list[BackwardNN]):
        self.models = models

    def forward(self, x: torch.Tensor):
        print(type(self.models))
        preds = [model.forward(x) for model in self.models]
        return torch.mean(torch.stack(preds), dim=0)
