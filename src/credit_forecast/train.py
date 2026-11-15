"""Deterministic CPU training with validation-only early stopping."""

import copy
import random
import time

import numpy as np
import torch
from torch import nn

from .model import CreditLSTM


def fit(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
        *, hidden: int, layers: int, dropout: float, learning_rate: float,
        batch_size: int, max_epochs: int, patience: int, seed: int):
    if min(len(x_train), len(x_val), batch_size, max_epochs, patience) < 1:
        raise ValueError("Training/validation and hyperparameters must be positive")
    # CPU-only avoids device-specific behavior; exact equality across PyTorch versions is not promised.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = CreditLSTM(hidden, layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    x_t, y_t = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_v, y_v = torch.from_numpy(x_val), torch.from_numpy(y_val)
    rng = np.random.default_rng(seed)
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in np.array_split(rng.permutation(len(x_t)), range(batch_size, len(x_t), batch_size)):
            optimizer.zero_grad()
            loss = criterion(model(x_t[batch]), y_t[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = float(criterion(model(x_t), y_t))
            val_loss = float(criterion(model(x_v), y_v))
        history.append({"epoch": epoch, "train_mse_scaled": train_loss, "val_mse_scaled": val_loss})
        if val_loss < best - 1e-8:
            best, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, history, best_epoch, time.perf_counter() - start


def predict(model: CreditLSTM, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy().reshape(-1)
