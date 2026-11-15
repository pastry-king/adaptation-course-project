"""Read and validate the audited monthly CSV without modifying its values."""

import csv
import hashlib
from datetime import date
from pathlib import Path

import numpy as np


def load_history(path: str | Path) -> tuple[list[date], np.ndarray, list[str]]:
    path = Path(path)
    months, amounts, methods = [], [], []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"month", "corporate_and_ip_mln_rub", "individual_mln_rub", "total_mln_rub", "source_method"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            month = date.fromisoformat(row["month"])
            corporate = float(row["corporate_and_ip_mln_rub"])
            individual = float(row["individual_mln_rub"])
            total = float(row["total_mln_rub"])
            if month.day != 1 or not np.isfinite([corporate, individual, total]).all():
                raise ValueError(f"Invalid month or value: {row['month']}")
            if min(corporate, individual, total) < 0 or total != corporate + individual:
                raise ValueError(f"Invalid component sum: {month}")
            months.append(month)
            amounts.append(total)
            methods.append(row["source_method"])
    if not months:
        raise ValueError("Empty history")
    for earlier, later in zip(months, months[1:]):
        if (later.year * 12 + later.month) - (earlier.year * 12 + earlier.month) != 1:
            raise ValueError(f"Nonconsecutive or unordered months: {earlier}, {later}")
    return months, np.asarray(amounts, dtype=np.float64), methods


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
