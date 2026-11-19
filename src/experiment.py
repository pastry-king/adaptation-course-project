"""Графический запуск полного эксперимента: python src/experiment.py."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from PyQt6.QtCore import QObject, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QSpinBox, QStatusBar, QStyle, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QProgressBar,
)

from lstm_core import BASELINE_LAGS, ROOT, RunConfig, run_experiment
from prepare_data import prepare_data


LABELS = {
    "lstm": "LSTM", "last_hour": "Предыдущий час",
    "yesterday": "Тот же час вчера", "last_week": "Тот же час неделю назад",
}
MODEL_COLORS = {
    "lstm": "#087e8b", "last_hour": "#a25486",
    "yesterday": "#cf8a25", "last_week": "#6578b5",
}


def _template_columns() -> list[str]:
    with (ROOT / "templates" / "experiments.csv").open(encoding="utf-8", newline="") as f:
        return next(csv.reader(f))


def _save_row(row: dict) -> None:
    """Сохранить статус/результаты запуска, не изменяя строки других ID."""
    path = ROOT / "experiments" / "experiments.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = _template_columns()
    records = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != columns:
                raise ValueError("Колонки experiments.csv не совпадают с templates/experiments.csv.")
            records = list(reader)
    saved = {name: row.get(name, "") for name in columns}
    for i, old in enumerate(records):
        if old["experiment_id"] == row["experiment_id"]:
            records[i] = saved
            break
    else:
        records.append(saved)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _software() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": version("numpy"), "pandas": version("pandas"),
        "torch": version("torch"), "sklearn": version("scikit-learn"),
        "matplotlib": version("matplotlib"), "pyqt": version("PyQt6"),
        "jinja2": version("Jinja2"),
    }


def _image(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def _plots(result: dict) -> list[dict]:
    plots = []
    metadata = result["metadata"]
    # Обзор всего ряда; для читаемости показан каждый двенадцатый час.
    fig, ax = plt.subplots(figsize=(11, 3.3))
    timeline = result["timeline"]
    ax.plot(timeline["timestamp"].iloc[::12], timeline["energy_kwh"].iloc[::12],
            color="#087e8b", linewidth=0.8)
    for name, color in (("validation", "#6578b5"), ("forecast", "#a25486")):
        ax.axvline(datetime.fromisoformat(metadata["splits"][name]["start"]),
                   color=color, linestyle="--", linewidth=1.4, label=f"Начало {name}")
    ax.set(ylabel="кВт·ч за час", title="Хронология и границы выборок")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(frameon=False)
    fig.tight_layout()
    plots.append({"key": "timeline", "title": "Ряд", "alt": "Потребление за час и границы train, validation и forecast", "caption": "Показан каждый двенадцатый час; вертикальные линии разделяют периоды.", "png": _image(fig)})

    fig, ax = plt.subplots(figsize=(11, 3.3))
    epochs = range(1, result["epochs_run"] + 1)
    ax.plot(epochs, result["history"]["train"], label="Train", color="#6578b5")
    ax.plot(epochs, result["history"]["validation"], label="Validation", color="#087e8b")
    ax.axvline(result["best_epoch"], color="#a25486", linestyle="--", label="Лучшая эпоха")
    ax.set(xlabel="Эпоха", ylabel="MSE в нормализованной шкале", title="Ход обучения")
    ax.legend(frameon=False)
    fig.tight_layout()
    plots.append({"key": "training_loss", "title": "Обучение", "alt": "Ошибка train и validation по эпохам", "caption": "Веса лучшей эпохи используются для обоих периодов проверки.", "png": _image(fig)})

    for stage, title in (("validation", "Validation"), ("forecast", "Forecast")):
        data = result["predictions"][stage]
        last = slice(-168, None)
        fig, ax = plt.subplots(figsize=(11, 3.6))
        ax.plot(data["timestamps"][last], data["actual"][last], label="Факт", color="#172941", linewidth=1.5)
        for key in LABELS:
            ax.plot(data["timestamps"][last], data[key][last], label=LABELS[key], color=MODEL_COLORS[key], linewidth=0.9, alpha=0.88)
        ax.set(ylabel="кВт·ч за час", title=f"{title}: последние 7 суток")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        ax.legend(ncol=3, fontsize=8, frameon=False)
        fig.tight_layout()
        plots.append({"key": f"{stage}_predictions", "title": f"Факт и прогнозы: {title}", "alt": f"Факт и четыре прогноза за последние семь суток {stage}", "caption": "Для каждого часа входом служат только предшествующие фактические часы.", "png": _image(fig)})
    data = result["predictions"]["forecast"]
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(data["timestamps"][-168:], (data["actual"] - data["lstm"])[-168:], color="#087e8b")
    ax.axhline(0, color="#6578b5", linestyle="--", linewidth=1)
    ax.set(ylabel="Факт − прогноз, кВт·ч", title="Ошибки LSTM на последней неделе forecast")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    fig.tight_layout()
    plots.append({"key": "forecast_errors", "title": "Ошибки", "alt": "Отклонения прогноза LSTM от факта", "caption": "Значение выше нуля означает недооценку потребления.", "png": _image(fig)})
    return plots


def _graph_tables(result: dict) -> dict[str, pd.DataFrame]:
    """Полные числовые ряды: из них можно восстановить графики и пересчитать метрики."""
    timeline = result["timeline"][["timestamp", "energy_kwh"]].copy()
    tables = {
        "timeline": timeline,
        "training_loss": pd.DataFrame({
            "epoch": range(1, result["epochs_run"] + 1),
            "train_mse_normalized": result["history"]["train"],
            "validation_mse_normalized": result["history"]["validation"],
            "is_best_epoch": [epoch == result["best_epoch"] for epoch in range(1, result["epochs_run"] + 1)],
        }),
    }
    for stage in ("validation", "forecast"):
        data = result["predictions"][stage]
        frame = pd.DataFrame({
            "timestamp": data["timestamps"],
            "actual_kwh": data["actual"],
            **{f"{name}_kwh": data[name] for name in LABELS},
        })
        for name in LABELS:
            frame[f"error_{name}_kwh"] = frame["actual_kwh"] - frame[f"{name}_kwh"]
        tables[f"{stage}_predictions"] = frame
    for table in tables.values():
        if "timestamp" in table:
            table["timestamp"] = pd.to_datetime(table["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return tables


def _save_graphs(plots: list[dict], report: dict, result: dict, root: Path = ROOT) -> list[dict]:
    """Сохранить графики, исходные для них таблицы и каталог по ID эксперимента."""
    experiment_id = report["experiment_id"]
    directory = root / "experiments" / "graphs" / experiment_id
    directory.mkdir(parents=True, exist_ok=False)
    report_path = root / report["artifacts"]["report_path"]
    data_dir = directory / "data"
    data_dir.mkdir()
    tables = {}
    for key, frame in _graph_tables(result).items():
        path = data_dir / f"{key}.csv"
        frame.to_csv(path, index=False, float_format="%.17g")
        tables[key] = {
            "file": path.relative_to(root).as_posix(),
            "rows": len(frame), "columns": list(frame.columns),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    plot_sources = {
        "timeline": ("timeline", {"mode": "stride", "step": 12}, ["energy_kwh"]),
        "training_loss": ("training_loss", {"mode": "all"}, ["train_mse_normalized", "validation_mse_normalized"]),
        "validation_predictions": ("validation_predictions", {"mode": "tail", "count": 168}, ["actual_kwh", *[f"{name}_kwh" for name in LABELS]]),
        "forecast_predictions": ("forecast_predictions", {"mode": "tail", "count": 168}, ["actual_kwh", *[f"{name}_kwh" for name in LABELS]]),
        "forecast_errors": ("forecast_predictions", {"mode": "tail", "count": 168}, ["error_lstm_kwh"]),
    }
    entries = []
    html_plots = []
    for plot in plots:
        table_key, selection, y_columns = plot_sources[plot["key"]]
        data_path = root / tables[table_key]["file"]
        filename = f"{plot['key']}.png"
        path = directory / filename
        path.write_bytes(plot["png"])
        project_path = path.relative_to(root).as_posix()
        entries.append({
            "key": plot["key"], "file": project_path,
            "title": plot["title"], "alt": plot["alt"], "caption": plot["caption"],
            "sha256": hashlib.sha256(plot["png"]).hexdigest(),
            "data_file": tables[table_key]["file"],
            "selection": selection,
            "x_column": "epoch" if table_key == "training_loss" else "timestamp",
            "y_columns": y_columns,
            "reference_lines": (
                [report["splits"][name]["start"] for name in ("validation", "forecast")]
                if plot["key"] == "timeline" else
                [report["training"]["best_epoch"]] if plot["key"] == "training_loss" else
                [0] if plot["key"] == "forecast_errors" else []
            ),
        })
        html_plots.append({
            "title": plot["title"], "alt": plot["alt"], "caption": plot["caption"],
            "file": os.path.relpath(path, report_path.parent).replace("\\", "/"),
            "data_file": os.path.relpath(data_path, report_path.parent).replace("\\", "/"),
            "data_uri": "data:image/png;base64," + base64.b64encode(plot["png"]).decode("ascii"),
        })
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "report_path": report["artifacts"]["report_path"],
        "time_zone": report["dataset"]["time_zone"],
        "units": {
            "energy_and_forecast": "кВт·ч за час",
            "errors": "кВт·ч; факт минус прогноз",
            "train_and_validation_loss": "MSE на нормализованных значениях",
        },
        "data_tables": tables,
        "graphs": entries,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    report["graphs"] = entries
    report["graph_data_tables"] = tables
    report["artifacts"]["graphs_manifest_path"] = (directory / "manifest.json").relative_to(root).as_posix()
    return html_plots


def _build_report(experiment_id: str, created_at: str, hypothesis: str, result: dict) -> dict:
    metadata = result["metadata"]
    metrics = result["metrics"]
    forecast = metrics["forecast"]
    best_naive = min(BASELINE_LAGS, key=lambda name: forecast[name]["mae"])
    delta = forecast[best_naive]["mae"] - forecast["lstm"]["mae"]
    conclusion = (
        f"MAE LSTM на forecast: {forecast['lstm']['mae']:.3f} кВт·ч; "
        f"лучший наивный прогноз — «{LABELS[best_naive]}», MAE {forecast[best_naive]['mae']:.3f} кВт·ч. "
        + ("LSTM точнее на этом периоде." if delta > 0 else "Наивный прогноз не уступил LSTM на этом периоде.")
    )
    config = asdict(result["config"])
    config["scaler"] = "StandardScaler (только train)"
    report = {
        "schema_version": 1, "experiment_id": experiment_id,
        "created_at": created_at, "status": "completed", "model_type": "LSTM",
        "hypothesis": hypothesis,
        "changed_factor": "window_size, num_layers, hidden_size",
        "factor_level": f"{config['window_size']}, {config['num_layers']}, {config['hidden_size']}",
        "controlled_conditions": f"Один ряд и границы выборок; seed={config['seed']}.",
        "dataset": metadata["dataset"], "splits": metadata["splits"],
        "configuration": config,
        "training": {
            "epochs_run": result["epochs_run"], "best_epoch": result["best_epoch"],
            "best_validation_loss": result["best_validation_loss"], "elapsed_seconds": result["elapsed_seconds"],
            "history": result["history"],
        },
        "metrics": metrics, "software": _software(),
        "artifacts": {
            "weights_path": f"experiments/weights/experiment_{experiment_id}.pt",
            "csv_path": "experiments/experiments.csv",
            "report_path": f"experiments/reports/experiment_{experiment_id}.html",
        },
        "conclusion": conclusion,
        "limitations": (
            "Один анонимный клиент; особенности перехода на летнее время; проверка на один час вперёд "
            "с известным прошлым, а не рекурсивный прогноз на весь сезон. "
            "Повторный подбор по forecast не даёт независимой оценки выбора лучшего запуска."
        ),
    }
    return report


def _csv_values(report: dict) -> dict:
    dataset = report["dataset"]
    config = report["configuration"]
    training = report["training"]
    software = report["software"]
    row = {
        "schema_version": report["schema_version"], "experiment_id": report["experiment_id"],
        "created_at": report["created_at"], "status": report["status"],
        "model_type": report["model_type"], "hypothesis": report["hypothesis"],
        "changed_factor": report["changed_factor"], "factor_level": report["factor_level"],
        "controlled_conditions": report["controlled_conditions"],
        "dataset_name": dataset["name"], "dataset_id": dataset["id"],
        "dataset_version": dataset["version"], "series_column": dataset["series_column"],
        "source_start": dataset["source_start"], "source_end": dataset["source_end"],
        "source_frequency": dataset["source_frequency"], "source_unit": dataset["source_unit"],
        "target_frequency": dataset["target_frequency"], "target_unit": dataset["target_unit"],
        "time_zone": dataset["time_zone"], "preprocessing": dataset["preprocessing"],
        "epochs_run": training["epochs_run"], "best_epoch": training["best_epoch"],
        "best_validation_loss": training["best_validation_loss"],
        "elapsed_seconds": training["elapsed_seconds"],
        "weights_path": report["artifacts"]["weights_path"],
        "report_path": report["artifacts"]["report_path"],
    }
    for stage, split in report["splits"].items():
        for field, target in (("start", "start"), ("end", "end"), ("points", "points")):
            row[f"{stage}_{target}"] = split[field]
    for key, value in config.items():
        row[key] = value
    for key, value in software.items():
        row[f"{key}_version"] = value
    for stage, models in report["metrics"].items():
        for name, scores in models.items():
            for metric, value in scores.items():
                row[f"{stage}_{name}_{metric}"] = value
    unknown = set(row) - set(_template_columns())
    if unknown:
        raise ValueError(f"В шаблоне CSV отсутствуют колонки: {sorted(unknown)}")
    return row


class ExperimentWorker(QObject):
    progress = pyqtSignal(str)
    epoch = pyqtSignal(int, float, float)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, config: RunConfig, hypothesis: str):
        super().__init__()
        self.config = config
        self.hypothesis = hypothesis

    def run(self) -> None:
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
        row = {
            "schema_version": 1, "experiment_id": experiment_id, "created_at": created,
            "status": "running", "model_type": "LSTM", "hypothesis": self.hypothesis,
            "window_size": self.config.window_size, "num_layers": self.config.num_layers,
            "hidden_size": self.config.hidden_size, "dropout": self.config.dropout,
            "learning_rate": self.config.learning_rate, "optimizer": self.config.optimizer,
            "batch_size": self.config.batch_size, "max_epochs": self.config.max_epochs,
            "patience": self.config.patience, "seed": self.config.seed, "device": self.config.device,
        }
        try:
            _save_row(row)
            self.progress.emit("Проверка данных; при необходимости загружается компактный Parquet…")
            prepare_data(ROOT, on_progress=self.progress.emit)
            self.progress.emit("Данные готовы. Начинается обучение.")
            weights = ROOT / "experiments" / "weights" / f"experiment_{experiment_id}.pt"
            result = run_experiment(self.config, weights, on_epoch=self.epoch.emit)
            report = _build_report(experiment_id, created, self.hypothesis, result)
            self.progress.emit("Обучение завершено. Создаются графики и отчёт.")
            plots = _save_graphs(_plots(result), report, result)
            html = Environment(
                loader=FileSystemLoader(ROOT / "templates"),
                autoescape=True, undefined=StrictUndefined,
            ).get_template("report.html.j2").render(report=report, plots=plots)
            report_path = ROOT / report["artifacts"]["report_path"]
            report_path.parent.mkdir(parents=True, exist_ok=True)
            if report_path.exists():
                raise FileExistsError(f"Отчёт этого ID уже существует: {report_path}")
            report_path.write_text(html, encoding="utf-8")
            _save_row(_csv_values(report))
            self.finished.emit({"id": experiment_id, "metrics": report["metrics"], "report_path": str(report_path)})
        except Exception as exc:
            row["status"] = "failed"
            row["error_message"] = f"{type(exc).__name__}: {exc}"
            try:
                _save_row(row)
            except Exception as save_error:
                self.failed.emit(f"{exc}\nТакже не удалось записать CSV: {save_error}")
                return
            self.failed.emit(str(exc))


class ExperimentWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Прогноз потребления · LSTM")
        self.resize(1080, 780)
        self.thread: QThread | None = None
        self.worker: ExperimentWorker | None = None
        self.last_report: str | None = None
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(16)

        title = QLabel("Прогноз потребления электроэнергии")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel("Один запуск: обучение → validation → forecast → веса, CSV и HTML-отчёт")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        columns = QHBoxLayout()
        layout.addLayout(columns)

        form_box = QGroupBox("Параметры эксперимента")
        form = QFormLayout(form_box)
        form.setSpacing(11)
        columns.addWidget(form_box, 1)
        self.window = self._integer(24, 1, 720)
        self.layers = self._integer(1, 1, 4)
        self.neurons = self._integer(32, 4, 256)
        self.dropout = self._decimal(0.2, 0.0, 0.9, 0.05, 2)
        self.learning_rate = self._decimal(0.001, 0.00001, 0.1, 0.0005, 5)
        self.optimizer = QComboBox()
        self.optimizer.addItems(["Adam", "AdamW", "SGD"])
        self.batch = self._integer(64, 1, 1024)
        self.epochs = self._integer(30, 1, 500)
        self.patience = self._integer(5, 1, 100)
        self.seed = self._integer(42, 0, 2_147_483_647)
        self.device = QComboBox()
        self.device.addItem("CPU", "cpu")
        try:
            import torch
            if torch.cuda.is_available():
                self.device.addItem("CUDA", "cuda")
        except ImportError:
            pass
        self.hypothesis = QLineEdit("Оценить прогноз LSTM с заданным окном и архитектурой")
        for label, field in (
            ("Окно, часов", self.window), ("Слоёв LSTM", self.layers),
            ("Нейронов в слое", self.neurons), ("Dropout", self.dropout),
            ("Learning rate", self.learning_rate), ("Оптимизатор", self.optimizer),
            ("Batch size", self.batch), ("Максимум эпох", self.epochs),
            ("Patience", self.patience), ("Seed", self.seed),
            ("Устройство", self.device), ("Гипотеза", self.hypothesis),
        ):
            form.addRow(label, field)

        right = QVBoxLayout()
        columns.addLayout(right, 1)
        summary = QGroupBox("Данные и этапы")
        summary_layout = QVBoxLayout(summary)
        for text in (
            "Источник: OpenML Electricity 46214 · один анонимный потребитель",
            "Цель: кВт·ч за следующий час · время по Португалии",
            "Train: январь 2013 — май 2014 · обучение весов",
            "Validation: июнь — август 2014 · выбор эпохи",
            "Forecast: сентябрь — декабрь 2014 · проверка модели",
        ):
            label = QLabel(text)
            label.setWordWrap(True)
            summary_layout.addWidget(label)
        right.addWidget(summary)
        result_box = QGroupBox("Результаты · MAE, кВт·ч")
        result_layout = QVBoxLayout(result_box)
        self.table = QTableWidget(4, 3)
        self.table.setHorizontalHeaderLabels(["Прогноз", "Validation", "Forecast"])
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        for row, name in enumerate(LABELS.values()):
            self.table.setItem(row, 0, QTableWidgetItem(name))
        result_layout.addWidget(self.table)
        right.addWidget(result_box)
        right.addStretch()

        buttons = QHBoxLayout()
        layout.addLayout(buttons)
        self.start_button = QPushButton("Запустить эксперимент")
        self.start_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.start_button.clicked.connect(self.start)
        buttons.addWidget(self.start_button)
        self.report_button = QPushButton("Открыть HTML-отчёт")
        self.report_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.report_button.setEnabled(False)
        self.report_button.clicked.connect(self.open_report)
        buttons.addWidget(self.report_button)
        buttons.addStretch()
        self.download_progress = QProgressBar()
        self.download_progress.setFormat("Загрузка исходных данных: %p%")
        self.download_progress.setVisible(False)
        layout.addWidget(self.download_progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        self.log.setPlaceholderText("Ход эксперимента появится здесь.")
        layout.addWidget(self.log, 1)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Готово к запуску · данные при необходимости загрузятся автоматически")
        self.setStyleSheet("""
            QMainWindow { background: #f2f5fa; color: #172941; }
            QLabel#title { font-size: 25px; font-weight: 700; color: #142c4d; }
            QLabel#subtitle { font-size: 13px; color: #586d84; }
            QGroupBox { background: white; border: 1px solid #dce4ed; border-radius: 12px;
                        margin-top: 10px; padding: 18px 14px 12px; font-weight: 650; }
            QGroupBox::title { subcontrol-origin: margin; left: 13px; padding: 0 6px; color: #193c5d; }
            QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox { background: #f8fafc; border: 1px solid #cdd9e4;
                border-radius: 6px; padding: 6px 8px; min-height: 22px; }
            QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus { border-color: #16838a; }
            QPushButton { border: 1px solid #b7ccd7; border-radius: 8px; background: white;
                padding: 9px 15px; font-weight: 600; }
            QPushButton:hover { background: #e6f4f4; }
            QPushButton:disabled { color: #8997a6; background: #edf0f3; }
            QPushButton:first-child { background: #166d7d; color: white; border-color: #166d7d; }
            QPlainTextEdit, QTableWidget { background: white; border: 1px solid #dce4ed;
                border-radius: 8px; padding: 7px; }
            QHeaderView::section { background: #eaf1f7; border: 0; padding: 7px; }
        """)

    @staticmethod
    def _integer(value: int, minimum: int, maximum: int) -> QSpinBox:
        box = QSpinBox()
        box.setRange(minimum, maximum)
        box.setValue(value)
        return box

    @staticmethod
    def _decimal(value: float, minimum: float, maximum: float, step: float, decimals: int) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setDecimals(decimals)
        box.setRange(minimum, maximum)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def start(self) -> None:
        config = RunConfig(
            window_size=self.window.value(), num_layers=self.layers.value(),
            hidden_size=self.neurons.value(), dropout=self.dropout.value(),
            learning_rate=self.learning_rate.value(), optimizer=self.optimizer.currentText(),
            batch_size=self.batch.value(), max_epochs=self.epochs.value(),
            patience=self.patience.value(), seed=self.seed.value(), device=self.device.currentData(),
        )
        try:
            config.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "Параметры", str(exc))
            return
        self.start_button.setEnabled(False)
        self.report_button.setEnabled(False)
        self.last_report = None
        self.log.clear()
        self.download_progress.setVisible(False)
        self.table.clearContents()
        for row, name in enumerate(LABELS.values()):
            self.table.setItem(row, 0, QTableWidgetItem(name))
        self.thread = QThread(self)
        self.worker = ExperimentWorker(config, self.hypothesis.text().strip() or "Проверить качество LSTM")
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.add_log)
        self.worker.epoch.connect(self.on_epoch)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.statusBar().showMessage("Эксперимент выполняется…")
        self.thread.start()

    def add_log(self, text: str) -> None:
        self.log.appendPlainText(text)
        if "Начало загрузки" in text:
            self.download_progress.setVisible(True)
            self.download_progress.setRange(0, 100 if "МиБ" in text else 0)
            self.download_progress.setValue(0)
        match = re.search(r"Скачано .*\((\d+)%\)", text)
        if match:
            self.download_progress.setVisible(True)
            self.download_progress.setRange(0, 100)
            self.download_progress.setValue(int(match.group(1)))
        if "Исходный файл сохранён" in text or "Данные готовы" in text:
            self.download_progress.setVisible(False)

    def on_epoch(self, epoch: int, train_loss: float, val_loss: float) -> None:
        self.add_log(f"Эпоха {epoch}: train MSE={train_loss:.5f}, validation MSE={val_loss:.5f}")

    def on_finished(self, outcome: dict) -> None:
        self.report_button.setEnabled(True)
        self.last_report = outcome["report_path"]
        for row, name in enumerate(LABELS):
            for col, stage in ((1, "validation"), (2, "forecast")):
                value = outcome["metrics"][stage][name]["mae"]
                self.table.setItem(row, col, QTableWidgetItem(f"{value:.3f}"))
        self.add_log(f"Эксперимент {outcome['id']} завершён. Отчёт: {self.last_report}")
        self.statusBar().showMessage(f"Завершён · {outcome['id']}")

    def on_failed(self, message: str) -> None:
        self.add_log(f"Ошибка: {message}")
        self.statusBar().showMessage("Ошибка эксперимента")
        QMessageBox.critical(self, "Эксперимент не завершён", message)

    def on_thread_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.thread = None
        self.worker = None

    def open_report(self) -> None:
        if self.last_report:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_report))

    def closeEvent(self, event) -> None:
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, "Эксперимент выполняется", "Дождитесь завершения обучения перед закрытием окна.")
            event.ignore()
        else:
            event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ExperimentWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
