"""Metrics in original units and headless plots for the experiment report."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or not actual.size or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and forecast must be equally shaped, nonempty and finite")
    error = actual - predicted
    return {"mae_mln_rub": float(np.mean(np.abs(error))),
            "mse_mln_rub_squared": float(np.mean(error ** 2))}


def save_history(months, values, methods, split, path: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(months, values / 1e6, color="#164e63", label="Выдачи, трлн руб.")
    ax.axvline(months[split.validation[0]], color="#ea580c", linestyle="--", label="Начало валидации")
    ax.axvline(months[split.test[0]], color="#b91c1c", linestyle="--", label="Начало теста")
    if len(set(methods)) > 1:
        boundary = next(i for i in range(1, len(methods)) if methods[i] != methods[i - 1])
        ax.axvline(months[boundary], color="#7c3aed", linestyle=":", label="Смена методики")
    ax.set(title="Месячный объём новых кредитов, РФ", ylabel="Трлн номинальных руб.", xlabel="Месяц")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_seasonality(months, values, methods, path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for method in dict.fromkeys(methods):
        subset = np.array([i for i, name in enumerate(methods) if name == method])
        # Normalise by each year's mean to compare seasonal shape, not price levels.
        years = np.array([months[i].year for i in subset])
        ratios = values[subset] / np.array([values[subset[years == y]].mean() for y in years])
        means = [np.mean(ratios[[months[i].month == m for i in subset]]) for m in range(1, 13)]
        label = "Архив (непроверенная сопоставимость)" if "archive" in method else "Современные таблицы"
        ax.plot(range(1, 13), means, marker="o", label=label)
    ax.axhline(1, color="gray", linewidth=0.8)
    ax.set(xticks=range(1, 13), xlabel="Месяц", ylabel="К среднему за свой год",
           title="Внутригодовая сезонность: описательное сравнение")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_comparison(months, actual, predictions, path: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(months, actual / 1e6, marker="o", color="#111827", label="Факт")
    for name, pred in predictions.items():
        ax.plot(months, pred / 1e6, marker=".", label=name)
    ax.set(title="Отложенный тест: прогноз на один месяц по известной истории",
           xlabel="Месяц выдачи", ylabel="Трлн номинальных руб.")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_training(history, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r["epoch"] for r in history], [r["train_mse_scaled"] for r in history], label="Обучение")
    ax.plot([r["epoch"] for r in history], [r["val_mse_scaled"] for r in history], label="Валидация")
    ax.set(xlabel="Эпоха", ylabel="MSE в стандартизованных единицах", title="Кривые обучения")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_screening(rows, path: Path):
    labels = [f"{r['window']} мес. / {r['hidden']}×{r['layers']}" for r in rows]
    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 0.65), 4))
    ax.bar(range(len(rows)), [r["val_mae_mln_rub"] / 1e6 for r in rows], color="#0891b2")
    ax.set(xticks=range(len(rows)), xticklabels=labels, ylabel="MAE, трлн руб.",
           title="Подбор только на валидации (ниже лучше)")
    ax.tick_params(axis="x", labelrotation=65)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
