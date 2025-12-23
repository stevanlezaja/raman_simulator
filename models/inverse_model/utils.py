import torch
import torch.nn as nn
from .random_projection_model import RandomProjectionInverseModel


def train_random_projection(X_train, Y_train, hidden_dim=1000, lam=1e-3):
    """
    X_train: tensor of size (num_samples, input_dim)
    Y_train: tensor of size (num_samples, output_dim)
    lam    : regularization factor
    """
    # Step 1: Random hidden layer
    W_hidden = torch.randn(X_train.shape[1], hidden_dim)
    b_hidden = torch.randn(hidden_dim)
    H = torch.tanh(X_train @ W_hidden + b_hidden)

    # Step 2: Solve for output weights analytically: beta = (H'H + λI)^(-1) H'Y
    HtH = H.T @ H
    reg = lam * torch.eye(hidden_dim)
    beta = torch.linalg.solve(HtH + reg, H.T @ Y_train)

    return W_hidden, b_hidden, beta
