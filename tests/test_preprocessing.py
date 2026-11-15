from datetime import date

import numpy as np
import pytest

from credit_forecast.baselines import forecasts
from credit_forecast.evaluate import metrics
from credit_forecast.preprocessing import Standardizer, chronological_split, make_windows


def test_windows_do_not_use_target_or_later_months():
    values = np.arange(16, dtype=float)
    x, y = make_windows(values, np.array([12, 13]), 6)
    np.testing.assert_array_equal(x[0, :, 0], [6, 7, 8, 9, 10, 11])
    np.testing.assert_array_equal(x[1, :, 0], [7, 8, 9, 10, 11, 12])
    np.testing.assert_array_equal(y[:, 0], [12, 13])
    naive = forecasts(values, np.array([12, 13]))
    np.testing.assert_array_equal(naive["year_ago"], [0, 1])
    np.testing.assert_array_equal(naive["previous_month"], [11, 12])


def test_train_scaler_is_not_fitted_on_validation_or_test():
    data = np.array([1., 2., 3., 1000., 2000.])
    scaler = Standardizer.fit(data[:3])
    assert scaler.mean == 2
    assert scaler.transform(data[3:])[0] > 100
    np.testing.assert_allclose(scaler.inverse(scaler.transform(data)), data)


def test_chronological_split_and_boundary_validation():
    months = [date(2020 + i // 12, i % 12 + 1, 1) for i in range(24)]
    result = chronological_split(months, months[0], months[11], months[17], months[-1])
    assert (len(result.train), len(result.validation), len(result.test)) == (12, 6, 6)
    assert result.train[-1] < result.validation[0] < result.test[0]
    with pytest.raises(ValueError):
        chronological_split(months, months[0], months[17], months[11], months[-1])


def test_metrics_are_in_original_units():
    result = metrics(np.array([10., 20.]), np.array([8., 16.]))
    assert result == {"mae_mln_rub": 3.0, "mse_mln_rub_squared": 10.0}
