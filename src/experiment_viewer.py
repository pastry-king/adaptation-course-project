"""Просмотр и безопасное удаление отдельных запусков: python src/experiment_viewer.py."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)


ROOT = Path(__file__).resolve().parents[1]
ID_PATTERN = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}\Z")
GRAPH_FILES = {
    "manifest.json", "timeline.png", "training_loss.png",
    "validation_predictions.png", "forecast_predictions.png", "forecast_errors.png",
}
GRAPH_DATA = {"timeline.csv", "training_loss.csv", "validation_predictions.csv", "forecast_predictions.csv"}


def read_records(root: Path = ROOT) -> tuple[list[str], list[dict]]:
    path = root / "experiments" / "experiments.csv"
    if not path.is_file():
        return [], []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames or [], list(reader)


def artifact_paths(root: Path, experiment_id: str) -> tuple[Path, Path, Path]:
    if not ID_PATTERN.fullmatch(experiment_id):
        raise ValueError("Неизвестный формат ID; удаление остановлено для безопасности.")
    directory = root / "experiments"
    return (
        directory / "weights" / f"experiment_{experiment_id}.pt",
        directory / "reports" / f"experiment_{experiment_id}.html",
        directory / "graphs" / experiment_id,
    )


def _check_owned_files(paths: tuple[Path, Path, Path]) -> None:
    """Не удалять симлинки и посторонние файлы внутри каталога графиков."""
    weights, report, graphs = paths
    for directory in (weights.parent.parent, weights.parent, report.parent, graphs.parent):
        if directory.is_symlink():
            raise ValueError(f"Обнаружена ссылка вместо каталога экспериментов: {directory}")
    for path in (weights, report, graphs):
        if path.is_symlink():
            raise ValueError(f"Обнаружена ссылка вместо файла эксперимента: {path}")
    for path in (weights, report):
        if path.exists() and not path.is_file():
            raise ValueError(f"Ожидался файл: {path}")
    if not graphs.exists():
        return  # старые эксперименты не имеют отдельного каталога графиков
    if not graphs.is_dir():
        raise ValueError(f"Ожидался каталог графиков: {graphs}")
    for item in graphs.iterdir():
        if item.is_symlink():
            raise ValueError(f"Обнаружена ссылка в каталоге графиков: {item}")
        if item.name == "data" and item.is_dir():
            for data_file in item.iterdir():
                if data_file.is_symlink() or not data_file.is_file() or data_file.name not in GRAPH_DATA:
                    raise ValueError(f"Неизвестный файл данных: {data_file}")
        elif not item.is_file() or item.name not in GRAPH_FILES:
            raise ValueError(f"В каталоге есть посторонний файл: {item}")


def delete_experiment(experiment_id: str, root: Path = ROOT) -> None:
    """Удалить одну запись и её артефакты; при ошибке записи CSV вернуть файлы."""
    root = Path(root)
    columns, records = read_records(root)
    selected = [row for row in records if row.get("experiment_id") == experiment_id]
    if len(selected) != 1:
        raise ValueError("Эксперимент не найден в журнале или ID повторяется.")
    if any(row.get("status") == "running" for row in records):
        raise ValueError("В журнале есть выполняющийся эксперимент. Дождитесь его завершения.")
    paths = artifact_paths(root, experiment_id)
    _check_owned_files(paths)
    journal = root / "experiments" / "experiments.csv"
    replacement = journal.with_name(f"experiments.{uuid4().hex}.tmp")
    staging = None
    moved: list[tuple[Path, Path]] = []
    try:
        with replacement.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(row for row in records if row["experiment_id"] != experiment_id)
        staging = Path(tempfile.mkdtemp(prefix=f".delete_{experiment_id}_", dir=journal.parent))
        for path in paths:
            if path.exists():
                destination = staging / path.name
                path.rename(destination)
                moved.append((path, destination))
        os.replace(replacement, journal)
    except Exception:
        for original, staged in reversed(moved):
            staged.rename(original)
        raise
    finally:
        replacement.unlink(missing_ok=True)
        if staging is not None and not any(staging.iterdir()):
            staging.rmdir()
    # Удаляется только временная папка, куда мы сами перенесли проверенные файлы этого ID.
    if staging is not None and moved:
        try:
            shutil.rmtree(staging)
        except OSError as exc:
            raise OSError(f"Запись удалена; файлы остались во временной папке {staging}: {exc}") from exc


class Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Эксперименты LSTM")
        self.resize(900, 620)
        panel = QWidget()
        self.setCentralWidget(panel)
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Выберите эксперимент для просмотра параметров, метрик и отчёта"))
        bar = QHBoxLayout()
        layout.addLayout(bar)
        self.choice = QComboBox()
        self.choice.currentIndexChanged.connect(self.show_selected)
        bar.addWidget(self.choice, 1)
        reload_button = QPushButton("Обновить")
        reload_button.clicked.connect(self.refresh)
        bar.addWidget(reload_button)
        self.open_button = QPushButton("Открыть HTML")
        self.open_button.clicked.connect(self.open_report)
        bar.addWidget(self.open_button)
        self.delete_button = QPushButton("Удалить эксперимент")
        self.delete_button.clicked.connect(self.delete_selected)
        bar.addWidget(self.delete_button)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details, 1)
        self.setStyleSheet("""
            QMainWindow { background: #f2f5fa; }
            QLabel { color: #193c5d; font-size: 16px; font-weight: 600; padding: 10px 0; }
            QComboBox, QPlainTextEdit { background: white; border: 1px solid #cdd9e4; border-radius: 7px; padding: 8px; }
            QPushButton { background: white; border: 1px solid #b7ccd7; border-radius: 7px; padding: 9px; }
            QPushButton:hover { background: #e6f4f4; }
        """)
        self.refresh()

    def refresh(self) -> None:
        current = self.choice.currentData()
        try:
            _, self.records = read_records()
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка журнала", str(exc))
            return
        self.choice.blockSignals(True)
        self.choice.clear()
        for row in self.records:
            experiment_id = row.get("experiment_id", "")
            self.choice.addItem(
                f"{experiment_id}  |  {row.get('status', '')}  |  "
                f"окно {row.get('window_size', '')}, слои {row.get('num_layers', '')}, нейроны {row.get('hidden_size', '')}",
                experiment_id,
            )
        if current:
            index = self.choice.findData(current)
            if index >= 0:
                self.choice.setCurrentIndex(index)
        self.choice.blockSignals(False)
        self.show_selected()

    def selected(self) -> dict | None:
        index = self.choice.currentIndex()
        return self.records[index] if 0 <= index < len(self.records) else None

    def show_selected(self) -> None:
        row = self.selected()
        self.open_button.setEnabled(bool(row))
        self.delete_button.setEnabled(bool(row))
        if row is None:
            self.details.setPlainText("Экспериментов пока нет.")
            return
        lines = [f"ЭКСПЕРИМЕНТ {row['experiment_id']}", "", "Все поля журнала:"]
        lines.extend(f"{name}: {value or '—'}" for name, value in row.items())
        try:
            report_path = artifact_paths(ROOT, row["experiment_id"])[1]
        except ValueError as exc:
            self.open_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            self.details.setPlainText("\n".join(lines + ["", f"Некорректный ID: {exc}"]))
            return
        if report_path.is_file():
            match = re.search(r'<script type="application/json" id="experiment-data">(.*?)</script>',
                              report_path.read_text(encoding="utf-8"), re.S)
            if match:
                try:
                    report = json.loads(match.group(1))
                    lines.extend(["", "Вывод из HTML:", report.get("conclusion", ""),
                                  "", "Ограничения:", report.get("limitations", "")])
                    if report.get("graphs"):
                        lines.extend(["", "Файлы графиков и данных:"])
                        lines.extend(f"{plot['key']}: {plot['file']} | {plot.get('data_file', 'данные не сохранены')}"
                                     for plot in report["graphs"])
                except (ValueError, KeyError) as exc:
                    lines.extend(["", f"Не удалось прочесть данные HTML-отчёта: {exc}"])
        self.details.setPlainText("\n".join(lines))

    def open_report(self) -> None:
        row = self.selected()
        if row:
            path = artifact_paths(ROOT, row["experiment_id"])[1]
            if not path.is_file() or not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
                QMessageBox.warning(self, "Отчёт", "HTML-отчёт не найден или не удалось открыть.")

    def delete_selected(self) -> None:
        row = self.selected()
        if row is None:
            return
        experiment_id = row["experiment_id"]
        reply = QMessageBox.question(
            self, "Подтверждение удаления",
            f"Безвозвратно удалить эксперимент {experiment_id}?\n\n"
            "Будут удалены строка CSV, веса, HTML и его каталог PNG/числовых данных. "
            "Общие данные в data/ не затрагиваются.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            delete_experiment(experiment_id)
        except Exception as exc:
            QMessageBox.critical(self, "Удаление не завершено", str(exc))
            return
        self.refresh()


def main() -> None:
    app = QApplication([])
    window = Viewer()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
