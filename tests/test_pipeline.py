import csv
from pathlib import Path

import pytest
import torch
import yaml

from credit_forecast.experiments import analyze, configuration, run
from credit_forecast.model import CreditLSTM
from credit_forecast.train import predict
import numpy as np


def test_small_end_to_end_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with (root / "configs/experiment.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config.update({"data_path": str(root / "data/processed/monthly_credit_history.csv"),
                   "output_dir": str(tmp_path / "results"), "artifact_dir": str(tmp_path / "artifacts"),
                   "scenarios": {"current_only": "2019-01-01"},
                   "windows": {"current_only": [12]}, "top_windows": 1,
                   "architectures": [{"hidden": 4, "layers": 1}], "screen_hidden": 4,
                   "final_seeds": [42], "max_epochs": 2, "patience": 1})
    config_path = tmp_path / "configs" / "experiment.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    cfg = configuration(config_path)
    analyze(cfg)
    run(cfg)
    result = tmp_path / "results/current_only"
    assert (tmp_path / "results/summary.csv").exists()
    assert (result / "test_forecasts.png").stat().st_size > 0
    assert (tmp_path / "results/analysis/history.png").stat().st_size > 0
    with (result / "test_predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert rows[0]["month"] == "2025-10-01"
    state = torch.load(tmp_path / "artifacts/current_only/seed_42.pt", map_location="cpu", weights_only=True)
    model = CreditLSTM(4, 1)
    model.load_state_dict(state)
    assert np.isfinite(predict(model, np.zeros((1, 12, 1), dtype=np.float32))).all()
    assert (tmp_path / "artifacts/current_only/seed_42.json").exists()
