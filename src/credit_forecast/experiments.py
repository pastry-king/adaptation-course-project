"""Predeclared window/architecture screening, independent test and artifacts."""

import csv
from datetime import date
import json
from pathlib import Path
import platform

import numpy as np
import matplotlib
import torch
import yaml

from .baselines import forecasts
from .data import load_history, sha256
from .evaluate import (metrics, save_comparison, save_history, save_screening,
                       save_seasonality, save_training)
from .preprocessing import Standardizer, chronological_split, make_windows
from .train import fit, predict


def configuration(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Expected a YAML mapping")
    root = path.parent.parent
    for field in ("data_path", "output_dir", "artifact_dir"):
        config[field] = (root / config[field]).resolve()
    for field in ("train_end", "validation_end", "test_end"):
        config[field] = date.fromisoformat(config[field])
    config["scenarios"] = {key: date.fromisoformat(value) for key, value in config["scenarios"].items()}
    if (not config["final_seeds"] or len(set(config["final_seeds"])) != len(config["final_seeds"])
            or config["top_windows"] < 1 or not config["architectures"]):
        raise ValueError("Provide unique final seeds, architectures and at least one selected window")
    return config


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No records to save")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def context(cfg: dict, name: str):
    months, values, methods = load_history(cfg["data_path"])
    if cfg["test_end"] != months[-1]:
        raise ValueError("test_end must be last published month for next-month forecast")
    split = chronological_split(months, cfg["scenarios"][name], cfg["train_end"],
                                cfg["validation_end"], cfg["test_end"])
    scaler = Standardizer.fit(values[split.train])
    return months, values, methods, split, scaler


def analyze(cfg: dict) -> None:
    """EDA, with provenance caveat: descriptive plots do not prove comparability."""
    months, values, methods = load_history(cfg["data_path"])
    out = cfg["output_dir"] / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    split = chronological_split(months, min(cfg["scenarios"].values()), cfg["train_end"],
                                cfg["validation_end"], cfg["test_end"])
    save_history(months, values, methods, split, out / "history.png")
    save_seasonality(months, values, methods, out / "seasonality.png")
    yoy = values[12:] / values[:-12] - 1
    jumps = sorted(range(len(yoy)), key=lambda i: abs(yoy[i]), reverse=True)[:8]
    text = ["# Первичный анализ ряда", "",
            f"CSV SHA-256: `{sha256(cfg['data_path'])}`. Единица: млн номинальных руб.",
            "", f"Период: {months[0]} — {months[-1]}, {len(months)} месяцев без разрывов.",
            "", "![История и границы](history.png)", "",
            "![Сезонность по методикам](seasonality.png)", "",
            "Месячная сезонность нормирована на среднее соответствующего календарного года; "
            "неполные годы представлены неполными годами. График не доказывает "
            "сопоставимость архивных разностей с современными месячными выдачами.", "",
            "## Наибольшие абсолютные изменения год к году", "",
            "| Месяц | Изменение к тому же месяцу год назад |", "| --- | ---: |"]
    text.extend(f"| {months[i + 12]} | {yoy[i]:+.1%} |" for i in jumps)
    text.extend(["", "Реальные скачки не удалены и не заменены. В январе 2019 сменились "
                 "формы и способ публикации, перекрытия для проверки нет. "
                 "Полный ряд — исследовательская гипотеза; отдельно проверяется период с 2019 года.", ""])
    (out / "README.md").write_text("\n".join(text), encoding="utf-8")


def _dataset(scaled: np.ndarray, split, window: int):
    train_idx = split.train[window:]
    if len(train_idx) < 12:
        raise ValueError(f"Only {len(train_idx)} training windows for window={window}; reduce window")
    return (make_windows(scaled, train_idx, window), make_windows(scaled, split.validation, window),
            make_windows(scaled, split.test, window))


def _train(cfg, datasets, scaler, values, indices, *, window, hidden, layers, seed):
    (x_train, y_train), (x_val, y_val), (x_test, _) = datasets
    model, history, epoch, elapsed = fit(x_train, y_train, x_val, y_val,
        hidden=hidden, layers=layers, dropout=cfg["dropout"],
        learning_rate=cfg["learning_rate"], batch_size=cfg["batch_size"],
        max_epochs=cfg["max_epochs"], patience=cfg["patience"], seed=seed)
    val_pred = scaler.inverse(predict(model, x_val))
    record = {"window": window, "hidden": hidden, "layers": layers, "seed": seed,
              "train_windows": len(x_train), "best_epoch": epoch, "epochs_run": len(history),
              "elapsed_seconds": round(elapsed, 3),
              "val_mae_mln_rub": metrics(values[indices.validation], val_pred)["mae_mln_rub"],
              "val_mse_mln_rub_squared": metrics(values[indices.validation], val_pred)["mse_mln_rub_squared"]}
    return model, history, record, x_test


def _report(name, cfg, months, values, split, screen, final, baseline_rows, next_rows, out):
    winner = min(screen, key=lambda r: (r["val_mae_mln_rub"], r["window"], r["hidden"], r["layers"]))
    lstm_mae = np.mean([r["test_mae_mln_rub"] for r in final])
    year_mae = next(r["mae_mln_rub"] for r in baseline_rows if r["model"] == "year_ago")
    lines = [f"# Эксперимент: {name}", "",
             f"Периоды: обучение {months[split.train[0]]} — {months[split.train[-1]]} ({len(split.train)}); "
             f"валидация {months[split.validation[0]]} — {months[split.validation[-1]]} ({len(split.validation)}); "
             f"тест {months[split.test[0]]} — {months[split.test[-1]]} ({len(split.test)}).",
             "", f"Данные SHA-256: `{sha256(cfg['data_path'])}`. Единицы: млн номинальных руб. "
             "MSE: (млн руб.)². Каждая тестовая точка — однократный прогноз по фактическим предыдущим месяцам, "
             "а не рекурсивное предсказание на 12 месяцев.", "",
             "## Подбор на валидации", "",
             "![Подбор конфигурации](screening.png)", "",
             "| Окно | Нейроны | Слои | MAE (млн руб.) | MSE ((млн руб.)²) |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    lines.extend(f"| {r['window']} | {r['hidden']} | {r['layers']} | {r['val_mae_mln_rub']:,.0f} | {r['val_mse_mln_rub_squared']:,.0f} |" for r in screen)
    lines.extend(["", f"Выбрано по минимуму validation MAE: окно {winner['window']}, "
                  f"{winner['hidden']} нейронов, {winner['layers']} слоёв. "
                  "Тест при выборе конфигурации не использовался.", "",
                  "## Независимый тест", "",
                  "| Метод | Seed | MAE (млн руб.) | MSE ((млн руб.)²) |",
                  "| --- | ---: | ---: | ---: |"])
    lines.extend(f"| {r['model']} | — | {r['mae_mln_rub']:,.0f} | {r['mse_mln_rub_squared']:,.0f} |" for r in baseline_rows)
    lines.extend(f"| LSTM | {r['seed']} | {r['test_mae_mln_rub']:,.0f} | {r['test_mse_mln_rub_squared']:,.0f} |" for r in final)
    lines.extend(["", f"LSTM: средняя MAE по заранее заданным seed {lstm_mae:,.0f}, "
                  f"разброс {min(r['test_mae_mln_rub'] for r in final):,.0f}–"
                  f"{max(r['test_mae_mln_rub'] for r in final):,.0f} млн руб. "
                  + ("Средняя MAE ниже прошлогоднего ориентира, но нужна проверка устойчивости по seed и методикам."
                     if lstm_mae < year_mae else "Средняя MAE не ниже прошлогоднего ориентира: преимущество не показано."),
                  "", "![Тест: факт и прогнозы](test_forecasts.png)", "",
                  "![Кривые обучения](training.png)", "",
                  "## Самые большие промахи LSTM (seed первого запуска)", "",
                  "| Месяц | Факт | Прогноз | Абсолютная ошибка (млн руб.) |",
                  "| --- | ---: | ---: | ---: |"])
    with (out / "test_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    for r in sorted(predictions, key=lambda r: abs(float(r["actual_mln_rub"]) - float(r["lstm_mln_rub"])), reverse=True)[:3]:
        actual, pred = float(r["actual_mln_rub"]), float(r["lstm_mln_rub"])
        lines.append(f"| {r['month']} | {actual:,.0f} | {pred:,.0f} | {abs(actual - pred):,.0f} |")
    lines.extend(["", "## Прогноз на месяц после конца локального ряда", "",
                  f"{next_rows[0]['month']}: LSTM по seed = " + ", ".join(f"{r['seed']}: {r['forecast_mln_rub']:,.0f}" for r in next_rows)
                  + " млн руб. Ориентир «прошлый год»: " + f"{values[-12]:,.0f}; "
                  + "ориентир «прошлый месяц»: " + f"{values[-1]:,.0f} млн руб. "
                   "Это прогноз **одного** месяца, для которого в использованном CSV нет факта; "
                   "перед использованием сверьте его с актуальной публикацией ЦБ.", "",
                  "**Ограничение:** архивные данные до 2019 года восстановлены из накопленных итогов; "
                  "сопоставимость со сменившейся методикой и влияние валютной переоценки не доказаны. "
                  f"Сравните вывод с [`{'current_only' if name == 'full' else 'full'}`](../{'current_only' if name == 'full' else 'full'}/README.md) "
                  "прежде, чем делать практический вывод. 12 тестовых точек и общие "
                  "экономические сдвиги ограничивают силу вывода.", ""])
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: dict) -> None:
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    cfg["artifact_dir"].mkdir(parents=True, exist_ok=True)
    conclusions = []
    for name in cfg["scenarios"]:
        print(f"Scenario: {name}", flush=True)
        months, values, methods, split, scaler = context(cfg, name)
        out = cfg["output_dir"] / name
        art = cfg["artifact_dir"] / name
        out.mkdir(parents=True, exist_ok=True)
        art.mkdir(parents=True, exist_ok=True)
        scaled = scaler.transform(values)
        datasets = {w: _dataset(scaled, split, w) for w in cfg["windows"][name]}
        screen = []
        # Stage 1: window effect with fixed architecture and seed.
        for window in cfg["windows"][name]:
            _, _, row, _ = _train(cfg, datasets[window], scaler, values, split,
                window=window, hidden=cfg["screen_hidden"], layers=1, seed=cfg["screen_seed"])
            screen.append({"stage": "window", **row})
            print(f"  window={window}, val MAE={row['val_mae_mln_rub']:.0f}", flush=True)
        winners = sorted(screen, key=lambda r: (r["val_mae_mln_rub"], r["window"]))[:cfg["top_windows"]]
        # Stage 2: architecture effect for preselected top windows only.
        for winner in winners:
            for arch in cfg["architectures"]:
                if arch["hidden"] == cfg["screen_hidden"] and arch["layers"] == 1:
                    continue  # Already measured in stage 1.
                _, _, row, _ = _train(cfg, datasets[winner["window"]], scaler, values, split,
                    window=winner["window"], hidden=arch["hidden"], layers=arch["layers"],
                    seed=cfg["screen_seed"])
                screen.append({"stage": "architecture", **row})
        write_csv(out / "validation_screening.csv", screen)
        save_screening(screen, out / "screening.png")
        selected = min(screen, key=lambda r: (r["val_mae_mln_rub"], r["window"], r["hidden"], r["layers"]))
        baseline = forecasts(values, split.test)
        baseline_rows = [{"model": label, **metrics(values[split.test], pred)} for label, pred in baseline.items()]
        write_csv(out / "baseline_metrics.csv", baseline_rows)
        final, next_rows = [], []
        next_month = date(months[-1].year + (months[-1].month == 12), months[-1].month % 12 + 1, 1)
        predictions = None
        for seed in cfg["final_seeds"]:
            model, history, record, x_test = _train(cfg, datasets[selected["window"]], scaler, values, split,
                window=selected["window"], hidden=selected["hidden"], layers=selected["layers"], seed=seed)
            test_pred = scaler.inverse(predict(model, x_test))
            row = {**record, **{f"test_{k}": v for k, v in metrics(values[split.test], test_pred).items()}}
            final.append(row)
            write_csv(out / f"test_predictions_seed_{seed}.csv", [
                {"month": str(months[i]), "actual_mln_rub": values[i], "lstm_mln_rub": test_pred[j]}
                for j, i in enumerate(split.test)])
            next_x = scaled[-selected["window"]:].astype(np.float32)[None, :, None]
            next_rows.append({"month": str(next_month), "seed": seed,
                              "forecast_mln_rub": float(scaler.inverse(predict(model, next_x))[0])})
            torch.save(model.state_dict(), art / f"seed_{seed}.pt")
            write_json(art / f"seed_{seed}.json", {
                "architecture": {"window": selected["window"], "hidden": selected["hidden"],
                                 "layers": selected["layers"], "dropout": cfg["dropout"]},
                "scaler": {"mean": scaler.mean, "std": scaler.std}, "seed": seed,
                "train_end": str(cfg["train_end"]), "validation_end": str(cfg["validation_end"]),
                "data_sha256": sha256(cfg["data_path"]), "best_epoch": record["best_epoch"]})
            write_csv(out / f"training_seed_{seed}.csv", history)
            if predictions is None:
                predictions = [{"month": str(months[i]), "actual_mln_rub": values[i],
                                "lstm_mln_rub": test_pred[j],
                                "year_ago_mln_rub": baseline["year_ago"][j],
                                "previous_month_mln_rub": baseline["previous_month"][j]}
                               for j, i in enumerate(split.test)]
                save_training(history, out / "training.png")
                save_comparison([months[i] for i in split.test], values[split.test],
                                {"LSTM (первый seed)": test_pred,
                                 "Прошлый год": baseline["year_ago"],
                                 "Прошлый месяц": baseline["previous_month"]}, out / "test_forecasts.png")
        write_csv(out / "test_predictions.csv", predictions)
        write_csv(out / "test_metrics.csv", final)
        write_csv(out / "next_month.csv", next_rows)
        write_json(out / "manifest.json", {
            "scenario": name, "data_path": str(cfg["data_path"]), "data_sha256": sha256(cfg["data_path"]),
            "raw_data_provenance": "data/README.md; source exports dated 2026-11-15",
            "train_start": str(months[split.train[0]]), "train_end": str(cfg["train_end"]),
            "validation_end": str(cfg["validation_end"]), "test_end": str(cfg["test_end"]),
            "train_count": len(split.train), "validation_count": len(split.validation),
            "test_count": len(split.test), "scaler": {"mean": scaler.mean, "std": scaler.std},
            "selected": {k: selected[k] for k in ("window", "hidden", "layers")},
            "seeds": cfg["final_seeds"], "screen_seed": cfg["screen_seed"],
            "training": {k: cfg[k] for k in ("dropout", "learning_rate", "batch_size", "max_epochs", "patience")},
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "torch": torch.__version__, "matplotlib": matplotlib.__version__,
                         "yaml": yaml.__version__}, "device": "cpu"})
        _report(name, cfg, months, values, split, screen, final, baseline_rows, next_rows, out)
        year_mae = next(r["mae_mln_rub"] for r in baseline_rows if r["model"] == "year_ago")
        conclusions.append({"scenario": name, "year_ago_mae_mln_rub": year_mae,
                            "lstm_mean_mae_mln_rub": float(np.mean([r["test_mae_mln_rub"] for r in final])),
                            "all_seeds_better": all(r["test_mae_mln_rub"] < year_mae for r in final)})
        print(f"  finished: {out / 'README.md'}", flush=True)
    write_csv(cfg["output_dir"] / "summary.csv", conclusions)
    robust = len(conclusions) > 1 and all(r["all_seeds_better"] for r in conclusions)
    lines = ["# Вывод по сценариям", "",
             "| Сценарий | MAE прошлогоднего, млн руб. | Средняя MAE LSTM, млн руб. | Все seed лучше? |",
             "| --- | ---: | ---: | --- |"]
    lines.extend(f"| [{r['scenario']}]({r['scenario']}/README.md) | {r['year_ago_mae_mln_rub']:,.0f} | "
                 f"{r['lstm_mean_mae_mln_rub']:,.0f} | {'Да' if r['all_seeds_better'] else 'Нет'} |"
                 for r in conclusions)
    conclusion = ("Проверка устойчивости между сценариями не выполнена."
                  if len(conclusions) < 2 else
                  "По этому критерию преимущество LSTM устойчиво к инициализации в обоих сценариях; "
                  "методологическую сопоставимость архива это всё равно не доказывает."
                  if robust else "Устойчивое преимущество LSTM одновременно по инициализациям и двум "
                  "сценариям не показано. Не следует выбирать лучший seed задним числом.")
    lines.extend(["", "**Итог:** " + conclusion,
                  "", "Вывод относится лишь к одношаговому прогнозу и конкретным 12 отложенным месяцам, "
                  "не к прибыли банков и не к многолетнему горизонту.", ""])
    (cfg["output_dir"] / "README.md").write_text("\n".join(lines), encoding="utf-8")
