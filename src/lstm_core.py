"""Скользящие окна, LSTM и оценка одного эксперимента (без GUI и отчётов)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
BASELINE_LAGS = {"last_hour": 1, "yesterday": 24, "last_week": 168}


@dataclass(frozen=True)
class RunConfig:
    window_size: int = 24
    num_layers: int = 1
    hidden_size: int = 32
    dropout: float = 0.2
    learning_rate: float = 0.001
    optimizer: str = "Adam"
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5
    seed: int = 42
    device: str = "cpu"

    def validate(self) -> None:
        if not 1 <= self.window_size <= 720:
            raise ValueError("Окно должно быть от 1 до 720 часов.")
        if not 1 <= self.num_layers <= 4 or not 4 <= self.hidden_size <= 256:
            raise ValueError("Слоёв должно быть 1–4, нейронов — 4–256.")
        if not 0 <= self.dropout < 1 or self.learning_rate <= 0:
            raise ValueError("Dropout должен быть в [0, 1), learning rate — положительным.")
        if self.optimizer not in ("Adam", "AdamW", "SGD"):
            raise ValueError("Неизвестный оптимизатор.")
        if min(self.batch_size, self.max_epochs, self.patience) < 1 or self.seed < 0:
            raise ValueError("Batch size, эпохи и patience должны быть положительными; seed >= 0.")
        if self.device not in ("cpu", "cuda") or (self.device == "cuda" and not torch.cuda.is_available()):
            raise ValueError("Выбранное устройство недоступно.")


class PowerLSTM(nn.Module):
    """Та же последовательность LSTM → последний шаг → Linear, что в учебном примере."""

    def __init__(self, config: RunConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=config.hidden_size,
            num_layers=config.num_layers, batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.fc(self.dropout(sequence[:, -1, :])).squeeze(-1)


def load_series(root: Path = ROOT) -> tuple[pd.DataFrame, dict, dict[str, slice]]:
    """Прочитать подготовленные части и проверить непрерывность/границы."""
    root = Path(root)
    metadata_path = root / "data" / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("Сначала подготовьте данные: python src/prepare_data.py")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata["dataset"]["id"] != 46214:
        raise ValueError("Неподдерживаемый набор данных или версия метаданных.")
    frames = []
    boundaries = {}
    position = 0
    for name in ("train", "validation", "forecast"):
        info = metadata["splits"][name]
        frame = pd.read_csv(root / info["path"], parse_dates=["timestamp"])
        if list(frame.columns) != ["timestamp", "energy_kwh"] or len(frame) != info["points"]:
            raise ValueError(f"Изменилась схема или размер data/{name}/hourly.csv.")
        if frame.empty or frame["timestamp"].iloc[0].isoformat() != info["start"] or frame["timestamp"].iloc[-1].isoformat() != info["end"]:
            raise ValueError(f"Изменены границы выборки {name}.")
        frames.append(frame)
        boundaries[name] = slice(position, position + len(frame))
        position += len(frame)
    series = pd.concat(frames, ignore_index=True)
    if series["timestamp"].isna().any() or series["timestamp"].duplicated().any():
        raise ValueError("Неправильные метки времени в выборках.")
    if not series["timestamp"].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Почасовые данные содержат пропущенные часы.")
    values = series["energy_kwh"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Есть пропущенные или отрицательные значения потребления.")
    return series, metadata, boundaries


def make_windows(values: np.ndarray, targets: np.ndarray, window_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Цель для индекса t, вход — только [t-window_size, ..., t-1]."""
    if len(targets) == 0 or targets[0] < window_size:
        raise ValueError("Недостаточно предыдущих часов для первого скользящего окна.")
    x = np.stack([values[t - window_size:t] for t in targets]).astype(np.float32)
    y = values[targets].astype(np.float32)
    return torch.from_numpy(x[:, :, None]), torch.from_numpy(y)


def _predict(model: PowerLSTM, x: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in x.split(512):
            chunks.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(chunks)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    scores = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }
    if not all(np.isfinite(value) for value in scores.values()):
        raise ValueError("Получена некорректная метрика; проверьте ряд и обучение.")
    return scores


def run_experiment(
    config: RunConfig,
    weights_path: Path,
    root: Path = ROOT,
    on_epoch: Callable[[int, float, float], None] | None = None,
) -> dict:
    """Один полный запуск: train → лучшая эпоха на validation → forecast.

    Часы validation/forecast до момента цели доступны как фактический контекст
    one-step-ahead. Эти периоды не используются для обновления весов.
    """
    config.validate()
    series, metadata, splits = load_series(root)
    original = series["energy_kwh"].to_numpy(dtype=np.float64)
    train_stop = splits["train"].stop
    if config.window_size >= train_stop:
        raise ValueError("Окно длиннее обучающей выборки.")
    if splits["validation"].start < max(BASELINE_LAGS.values()):
        raise ValueError("Для недельного прогноза недостаточно истории.")
    if np.std(original[:train_stop]) == 0:
        raise ValueError("Обучающий ряд постоянный: нормализация невозможна.")
    scaler = StandardScaler().fit(original[:train_stop, None])
    scaled = scaler.transform(original[:, None]).ravel()
    indices = {
        "train": np.arange(config.window_size, train_stop),
        "validation": np.arange(splits["validation"].start, splits["validation"].stop),
        "forecast": np.arange(splits["forecast"].start, splits["forecast"].stop),
    }
    x_train, y_train = make_windows(scaled, indices["train"], config.window_size)
    x_val, y_val = make_windows(scaled, indices["validation"], config.window_size)
    x_forecast, _ = make_windows(scaled, indices["forecast"], config.window_size)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(config.device)
    model = PowerLSTM(config).to(device)
    optimizer_type = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW, "SGD": torch.optim.SGD}[config.optimizer]
    optimizer = optimizer_type(model.parameters(), lr=config.learning_rate)
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_epoch = 0
    best_weights = None
    stale = 0
    history = {"train": [], "validation": []}
    started = perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item() * len(xb)
        train_loss = total / len(y_train)
        val_pred_scaled = _predict(model, x_val, device)
        val_loss = float(np.mean((val_pred_scaled - y_val.numpy()) ** 2))
        if not np.isfinite(val_loss):
            raise ValueError("Во время обучения получена некорректная ошибка validation.")
        history["train"].append(train_loss)
        history["validation"].append(val_loss)
        if val_loss < best_loss - 1e-8:
            best_loss, best_epoch = val_loss, epoch
            best_weights = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if on_epoch:
            on_epoch(epoch, train_loss, val_loss)
        if stale >= config.patience:
            break

    assert best_weights is not None
    model.load_state_dict(best_weights)
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_path.exists():
        raise FileExistsError(f"Веса этого эксперимента уже существуют: {weights_path}")
    torch.save({
        "state_dict": best_weights, "config": asdict(config), "best_epoch": best_epoch,
        "scaler_mean": float(scaler.mean_[0]), "scaler_scale": float(scaler.scale_[0]),
        "dataset_id": metadata["dataset"]["id"], "series_column": metadata["dataset"]["series_column"],
    }, weights_path)

    metrics = {}
    predictions = {}
    for stage, x in (("validation", x_val), ("forecast", x_forecast)):
        target = indices[stage]
        actual = original[target]
        lstm = scaler.inverse_transform(_predict(model, x, device)[:, None]).ravel()
        stage_predictions = {"lstm": lstm}
        stage_predictions.update({name: original[target - lag] for name, lag in BASELINE_LAGS.items()})
        metrics[stage] = {name: _metrics(actual, pred) for name, pred in stage_predictions.items()}
        predictions[stage] = {
            "timestamps": series["timestamp"].iloc[target].to_numpy(),
            "actual": actual,
            **stage_predictions,
        }
    return {
        "metadata": metadata, "config": config, "metrics": metrics, "predictions": predictions,
        "timeline": series[["timestamp", "energy_kwh"]],
        "history": history, "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "epochs_run": len(history["train"]), "elapsed_seconds": perf_counter() - started,
    }
