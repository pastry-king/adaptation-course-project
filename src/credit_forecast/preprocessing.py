"""Chronological splits, train-only scaling and causal one-step windows."""

from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def chronological_split(months: list[date], start: date, train_end: date,
                        validation_end: date, test_end: date) -> Split:
    if not start <= train_end < validation_end < test_end:
        raise ValueError("Expected start <= train_end < validation_end < test_end")
    if months[0] > start or months[-1] < test_end:
        raise ValueError("Configured months absent from CSV")
    for boundary in (start, train_end, validation_end, test_end):
        if boundary not in months:
            raise ValueError(f"Split boundary missing: {boundary}")
    train = np.array([i for i, m in enumerate(months) if start <= m <= train_end])
    val = np.array([i for i, m in enumerate(months) if train_end < m <= validation_end])
    test = np.array([i for i, m in enumerate(months) if validation_end < m <= test_end])
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Empty split")
    return Split(train, val, test)


@dataclass(frozen=True)
class Standardizer:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("Need at least two finite training values")
        std = float(np.std(values))
        if std == 0:
            raise ValueError("Training series is constant")
        return cls(float(np.mean(values)), std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * self.std + self.mean


def make_windows(values: np.ndarray, targets: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if window < 1 or not len(targets) or np.any(np.diff(targets) != 1):
        raise ValueError("Targets must be a nonempty, consecutive interval; window >= 1")
    if targets[0] < window or targets[-1] >= len(values):
        raise ValueError("Insufficient past or future target beyond series")
    x = np.stack([values[i - window:i] for i in targets]).astype(np.float32)
    y = values[targets].astype(np.float32)
    return x[:, :, None], y[:, None]
