"""Загрузка и подготовка одного почасового ряда Electricity (OpenML 46214).

Запуск: python src/prepare_data.py
Данные сохраняются один раз; существующие пользовательские файлы не перезаписываются.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = 46214
DATASET_VERSION = 15
SERIES_COLUMN = "value_0"
PARQUET_URL = "https://data.openml.org/datasets/0004/46214/dataset_46214.pq"
PARQUET_NAME = "electricity_46214.parquet"
START = pd.Timestamp("2013-01-01 00:00:00")
END_EXCLUSIVE = pd.Timestamp("2015-01-01 00:00:00")
TRAIN_END = pd.Timestamp("2014-06-01 00:00:00")
VALIDATION_END = pd.Timestamp("2014-09-01 00:00:00")
RAW_NAME = "electricity_value_0_15min.csv"
PROCESSED_NAME = "electricity_value_0_hourly.csv"
SPLIT_NAME = "hourly.csv"


def _check_regular(frame: pd.DataFrame, frequency: str, expected: pd.DatetimeIndex) -> None:
    if frame["timestamp"].isna().any() or frame["timestamp"].duplicated().any():
        raise ValueError("Есть пустые или повторяющиеся временные метки.")
    if not frame["timestamp"].reset_index(drop=True).equals(pd.Series(expected, name="timestamp")):
        raise ValueError(f"Временной ряд не покрывает весь интервал с частотой {frequency}.")
    values = frame.iloc[:, 1].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("В данных есть пропуски, бесконечности или отрицательные значения.")
    if np.std(values) == 0:
        raise ValueError("У выбранного клиента постоянное значение: прогнозировать нечего.")


def _paths(root: Path) -> dict[str, Path]:
    data = root / "data"
    return {
        "source": data / "source" / PARQUET_NAME,
        "raw": data / "raw" / RAW_NAME,
        "processed": data / "processed" / PROCESSED_NAME,
        "train": data / "train" / SPLIT_NAME,
        "validation": data / "validation" / SPLIT_NAME,
        "forecast": data / "forecast" / SPLIT_NAME,
        "metadata": data / "metadata.json",
        "readme": data / "README.md",
    }


def _notify(message: str, on_progress: Callable[[str], None] | None) -> None:
    text = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    if on_progress is None:
        try:
            print(text, flush=True)
        except UnicodeEncodeError:
            # В некоторых Windows-консолях кодировка вывода не поддерживает кириллицу.
            stream = sys.stdout
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
            print(text, flush=True)
    else:
        on_progress(text)


def _download_parquet(path: Path, on_progress: Callable[[str], None] | None) -> None:
    """Скачать компактный исходник; .part позволяет продолжить при поддержке HTTP Range."""
    if path.is_file():
        _notify(f"Исходный Parquet уже есть: {path} ({path.stat().st_size / 2**20:.1f} МиБ).", on_progress)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    resume = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "lstm-mini-project/1.0"}
    if resume:
        headers["Range"] = f"bytes={resume}-"
        _notify(f"Найдено {resume / 2**20:.1f} МиБ незавершённой загрузки; пробуем продолжить.", on_progress)
    _notify(f"Подключаемся к OpenML: {PARQUET_URL}", on_progress)
    try:
        response = urlopen(Request(PARQUET_URL, headers=headers), timeout=60)
        with response:
            continuing = resume > 0 and response.status == 206
            if resume and not continuing:
                _notify("Сервер не поддержал продолжение; скачиваем Parquet заново.", on_progress)
            start = resume if continuing else 0
            content_range = response.headers.get("Content-Range", "")
            total = int(content_range.rsplit("/", 1)[1]) if continuing and "/" in content_range and content_range.rsplit("/", 1)[1].isdigit() else 0
            if not total and response.headers.get("Content-Length"):
                total = start + int(response.headers["Content-Length"])
            _notify(f"Начало загрузки: {total / 2**20:.1f} МиБ." if total else "Начало загрузки; общий размер сервер не сообщил.", on_progress)
            downloaded = start
            last_update = time.monotonic()
            last_percent = -5
            with partial.open("ab" if continuing else "wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
                    percent = int(downloaded * 100 / total) if total else 0
                    now = time.monotonic()
                    if (total and percent >= last_percent + 5) or now - last_update >= 4:
                        amount = f"{downloaded / 2**20:.1f} / {total / 2**20:.1f} МиБ ({percent}%)" if total else f"{downloaded / 2**20:.1f} МиБ"
                        _notify(f"Скачано {amount}", on_progress)
                        last_update, last_percent = now, percent
            if total and downloaded != total:
                raise OSError(f"Загрузка прервалась: получено {downloaded} из {total} байт. Повторный запуск попробует продолжить.")
    except Exception as exc:
        raise OSError(
            f"Не удалось скачать Parquet с OpenML: {exc}. Можно скачать файл вручную "
            f"по ссылке {PARQUET_URL} и положить его в {path}."
        ) from exc
    partial.replace(path)
    _notify(f"Исходный файл сохранён: {path} ({path.stat().st_size / 2**20:.1f} МиБ).", on_progress)


def _source_frame(source_path: Path, on_progress: Callable[[str], None] | None) -> pd.DataFrame:
    _download_parquet(source_path, on_progress)
    _notify("Читаем только столбцы date и value_0 из Parquet…", on_progress)
    try:
        source = pd.read_parquet(source_path, columns=["date", SERIES_COLUMN])
    except Exception as exc:
        raise ValueError(f"Не удалось прочесть {source_path}. Проверьте файл Parquet и установленный pyarrow: {exc}") from exc
    _notify(f"Получено {len(source):,} строк (другие столбцы в память не загружались).", on_progress)
    dates = pd.to_datetime(source["date"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    mask = (dates >= START) & (dates < END_EXCLUSIVE)
    return pd.DataFrame({
        "timestamp": dates.loc[mask].to_numpy(),
        "power_kw": pd.to_numeric(source.loc[mask, SERIES_COLUMN], errors="raise").to_numpy(),
    }).sort_values("timestamp").reset_index(drop=True)


def prepare_data(root: Path = ROOT, on_progress: Callable[[str], None] | None = None) -> dict:
    """Подготовить данные или проверить готовый комплект; вернуть metadata.json."""
    root = Path(root)
    paths = _paths(root)
    outputs = [paths[name] for name in ("processed", "train", "validation", "forecast", "metadata")]
    if all(path.is_file() for path in [paths["raw"], *outputs]):
        _notify("Данные уже подготовлены; повторное скачивание не требуется.", on_progress)
        return json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if any(path.exists() for path in outputs):
        raise FileExistsError("Найден неполный набор data/. Проверьте его вручную: файлы не перезаписываются.")

    if paths["raw"].is_file():
        _notify("Читаем сохранённый 15-минутный ряд; скачивание не требуется.", on_progress)
        raw = pd.read_csv(paths["raw"], parse_dates=["timestamp"])
    else:
        raw = _source_frame(paths["source"], on_progress)
    if list(raw.columns) != ["timestamp", "power_kw"]:
        raise ValueError("Ожидаются столбцы timestamp,power_kw в сыром ряду.")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="raise")
    expected_raw = pd.date_range(START, END_EXCLUSIVE - pd.Timedelta(minutes=15), freq="15min")
    _check_regular(raw, "15 минут", expected_raw)
    _notify(f"Проверен 15-минутный ряд: {len(raw):,} значений за 2013–2014 годы.", on_progress)
    if len(raw) > 100_000:
        raise ValueError("Выбранный одномерный ряд превышает 100 тыс. значений.")

    # В исходнике метки местного времени без часового пояса: на дни перевода
    # часов приходится по 96 точек; сохраняем правило источника, не вызываем tz_localize.
    energy = raw.set_index("timestamp")["power_kw"].resample("h").sum(min_count=4) * 0.25
    hourly = energy.rename("energy_kwh").reset_index()
    expected_hourly = pd.date_range(START, END_EXCLUSIVE - pd.Timedelta(hours=1), freq="h")
    _check_regular(hourly, "1 час", expected_hourly)
    _notify(f"Получен почасовой ряд: {len(hourly):,} значений в кВт·ч.", on_progress)
    split_frames = {
        "train": hourly.loc[hourly["timestamp"] < TRAIN_END],
        "validation": hourly.loc[(hourly["timestamp"] >= TRAIN_END) & (hourly["timestamp"] < VALIDATION_END)],
        "forecast": hourly.loc[hourly["timestamp"] >= VALIDATION_END],
    }
    if any(frame.empty for frame in split_frames.values()):
        raise ValueError("Одна из выборок оказалась пустой.")
    _notify("Размеры выборок: " + ", ".join(f"{name} — {len(frame):,}" for name, frame in split_frames.items()), on_progress)

    def split_info(frame: pd.DataFrame, path: Path) -> dict:
        return {
            "start": frame["timestamp"].iloc[0].isoformat(),
            "end": frame["timestamp"].iloc[-1].isoformat(),
            "points": len(frame),
            "path": str(path.relative_to(root).as_posix()),
        }

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Electricity Load Diagrams 2011-2014",
            "id": DATASET_ID,
            "version": DATASET_VERSION,
            "url": "https://www.openml.org/d/46214",
            "download_url": PARQUET_URL,
            "download_format": "Parquet (только date и value_0 читаются в память)",
            "original_url": "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014",
            "series_column": SERIES_COLUMN,
            "location": "Португалия; точное местоположение клиента не раскрыто",
            "time_zone": "Местное португальское время; метки без UTC-смещения",
            "source_start": START.isoformat(),
            "source_end": (END_EXCLUSIVE - pd.Timedelta(minutes=15)).isoformat(),
            "source_frequency": "15 минут",
            "source_unit": "кВт",
            "target_frequency": "1 час",
            "target_unit": "кВт·ч",
            "preprocessing": (
                "Каждые четыре четвертьчасовых показания мощности суммируются с множителем "
                "0,25 часа. Пропуски не заполняются: при их обнаружении подготовка прерывается. "
                "Особенности дней перевода часов сохранены по правилам исходного набора."
            ),
        },
        "quality": {
            "raw_points": len(raw),
            "hourly_points": len(hourly),
            "zero_fraction_raw": float((raw["power_kw"] == 0).mean()),
            "zero_fraction_hourly": float((hourly["energy_kwh"] == 0).mean()),
            "min_energy_kwh": float(hourly["energy_kwh"].min()),
            "max_energy_kwh": float(hourly["energy_kwh"].max()),
            "dst_note": "Весной в исходнике часть интервала 01:00–02:00 равна нулю, осенью часть суммирована; сутки всё равно имеют 96 отметок.",
        },
        "splits": {name: split_info(frame, paths[name]) for name, frame in split_frames.items()},
        "files": {name: str(paths[name].relative_to(root).as_posix()) for name in ("raw", "processed")},
    }

    for key in ("raw", "processed", "train", "validation", "forecast", "metadata"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    # Не скрывать случайное пересоздание существующих данных без метаданных.
    if not paths["raw"].exists():
        _notify("Сохраняем сырой выбранный ряд в data/raw/…", on_progress)
        raw.to_csv(paths["raw"], index=False)
    _notify("Сохраняем почасовой ряд и три выборки…", on_progress)
    hourly.to_csv(paths["processed"], index=False)
    for name, frame in split_frames.items():
        frame.to_csv(paths[name], index=False)
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if not paths["readme"].exists():
        paths["readme"].write_text(
            "# Данные мини-проекта\n\n"
            "Источник: OpenML Electricity, ID 46214, версия 15; первоисточник UCI Electricity Load Diagrams 2011–2014.\n"
            "Один анонимный клиент (`value_0`), 2013–2014, время по Португалии; точный регион неизвестен.\n\n"
            "`raw/` — 15-минутные показания мощности (кВт); `processed/` — энергия за час (кВт·ч), "
            "сумма четырёх мощностей × 0,25 ч. `train/`, `validation/`, `forecast/` — "
            "хронологические части подготовленного ряда (январь 2013 — май 2014, "
            "июнь — август 2014, сентябрь — декабрь 2014). `forecast` — отложенный тест.\n\n"
            "Исходник искусственно сохраняет 96 отметок и на дни перевода часов; весной часть значений "
            "нулевая, осенью — суммированная. Заполнение пропусков не выполняется.\n"
            "Компактный исходник Parquet при загрузке находится в `source/`; в память читаются "
            "только столбцы `date` и `value_0`.\n"
            "Точные даты, числа точек, единицы и правила обработки: `metadata.json` (UTF-8).\n",
            encoding="utf-8",
        )
    _notify(f"Подготовка завершена. Метаданные: {paths['metadata']}", on_progress)
    return metadata


if __name__ == "__main__":
    info = prepare_data()
    print("Данные подготовлены:", ROOT / "data" / "metadata.json")
    print("Часов:", info["quality"]["hourly_points"])
