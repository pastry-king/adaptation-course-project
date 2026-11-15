"""Forecast each month from a previously observed month (no fitted parameters)."""

import numpy as np


def forecasts(values: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    if len(targets) == 0 or targets.min() < 12 or targets.max() >= len(values):
        raise ValueError("Both naive forecasts need 12 prior observed months")
    return {"previous_month": values[targets - 1], "year_ago": values[targets - 12]}
