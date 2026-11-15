from datetime import date
from pathlib import Path

import pytest

from credit_forecast.data import load_history


def test_project_history_is_contiguous_and_has_both_methods():
    path = Path(__file__).resolve().parents[1] / "data/processed/monthly_credit_history.csv"
    months, values, methods = load_history(path)
    assert (months[0], months[-1], len(months)) == (date(2009, 4, 1), date(2026, 9, 1), 210)
    assert len(set(methods)) == 2
    assert values[-1] == 10_155_547


def test_rejects_missing_month_and_inconsistent_sum(tmp_path):
    p = tmp_path / "sample.csv"
    header = "month,corporate_and_ip_mln_rub,individual_mln_rub,total_mln_rub,source_method\n"
    p.write_text(header + "2020-01-01,1,2,3,x\n2020-03-01,2,3,5,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Nonconsecutive"):
        load_history(p)
    p.write_text(header + "2020-01-01,1,2,4,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="component sum"):
        load_history(p)
