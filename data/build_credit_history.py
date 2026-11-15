"""Build the CBR monthly loan-origination history from the downloaded workbooks.

Requires openpyxl. Run: python data/build_credit_history.py
Source files live in data/raw; if absent, they are downloaded from the URLs below.
"""

import csv
import re
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUTPUT = ROOT / "processed" / "monthly_credit_history.csv"
SOURCES = {
    "corporate_current": (
        "01_04_D_New_loans_subj.xlsx",
        "https://www.cbr.ru/vfs/statistics/BankSector/Loans_to_corporations/01_04_D_New_loans_subj.xlsx",
    ),
    "individual_current": (
        "02_04_New_loans_ind.xlsx",
        "https://www.cbr.ru/vfs/statistics/BankSector/Mortgage/02_04_New_loans_ind.xlsx",
    ),
    "corporate_archive": (
        "302-01M.xlsx",
        "https://www.cbr.ru/Queries/StatTable/Excel/302-01M?lang=ru-RU",
    ),
    "individual_archive": (
        "302-02M.xlsx",
        "https://www.cbr.ru/Queries/StatTable/Excel/302-02M?lang=ru-RU",
    ),
}
MONTHS = {
    name: i for i, name in enumerate(
        "Январь Февраль Март Апрель Май Июнь Июль Август Сентябрь Октябрь Ноябрь Декабрь".split(), 1
    )
}


def source(key):
    filename, url = SOURCES[key]
    path = RAW / filename
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=120) as response:
            path.write_bytes(response.read())
    return path


def current(key):
    book = load_workbook(source(key), read_only=True, data_only=True)
    sheet = book["итого"]
    rows = sheet.iter_rows(values_only=True)
    header = next(row for row in rows if any(
        isinstance(value, str) and re.fullmatch(r"[А-Яа-яЁё]+ 20\d\d", value)
        for value in row[1:]
    ))
    total = next(row for row in rows if row[0] == "РОССИЙСКАЯ ФЕДЕРАЦИЯ")
    result = {}
    for label, value in zip(header[1:], total[1:]):
        assert isinstance(label, str) and isinstance(value, (int, float)), (key, label, value)
        name, year = label.split()
        month = date(int(year), MONTHS[name], 1)
        assert month not in result
        result[month] = int(value)
    book.close()
    return result


def archive(key):
    book = load_workbook(source(key), read_only=True, data_only=True)
    cumulative = {}
    for sheet in book:
        match = re.search(r"(\d{2})\.(\d{2})\.(\d{4})$", sheet.title)
        assert match, sheet.title
        day, month, year = map(int, match.groups())
        assert day == 1
        report_date = date(year, month, day)
        # B = all currencies; C = rouble loans; E = outstanding debt (NOT used).
        row = next(row for row in sheet.iter_rows(min_row=1, max_col=4, values_only=True)
                   if row[0] == "РОССИЙСКАЯ ФЕДЕРАЦИЯ")
        assert isinstance(row[1], (int, float)) and isinstance(row[2], (int, float))
        cumulative[report_date] = (int(row[1]), int(row[2]))
    book.close()
    result = {}
    for report_date, values in cumulative.items():
        # On the first of month M, the cumulative total includes up to M-1.
        year, month = (report_date.year - 1, 12) if report_date.month == 1 else (
            report_date.year, report_date.month - 1
        )
        observed = date(year, month, 1)
        if month == 1:
            monthly = values  # annual reset: do not subtract previous December
        else:
            previous = date(year, month, 1)
            if previous not in cumulative:
                continue  # Jan-Mar 2009 cannot be reconstructed separately
            monthly = tuple(v - p for v, p in zip(values, cumulative[previous]))
        assert all(v >= 0 for v in monthly), (key, observed, monthly)
        result[observed] = monthly
    return result


def main():
    old_corp, old_ind = archive("corporate_archive"), archive("individual_archive")
    new_corp, new_ind = current("corporate_current"), current("individual_current")
    assert old_corp.keys() == old_ind.keys()
    assert new_corp.keys() == new_ind.keys()
    assert len(old_corp) == 117 and len(new_corp) == 93
    assert not (old_corp.keys() & new_corp.keys())
    months = sorted(old_corp.keys() | new_corp.keys())
    assert months[0] == date(2009, 4, 1) and months[-1] == date(2026, 9, 1)
    assert all((right.year - left.year) * 12 + right.month - left.month == 1
               for left, right in zip(months, months[1:]))
    # Sanity check only: nonnegative foreign-currency differences do not prove
    # that exchange-rate effects are absent from the older all-currency series.
    assert all(corp[0] >= corp[1] and old_ind[month][0] >= old_ind[month][1]
               for month, corp in old_corp.items())
    assert old_corp[date(2018, 12, 1)][0] + old_ind[date(2018, 12, 1)][0] == 6356575
    assert new_corp[date(2026, 9, 1)] + new_ind[date(2026, 9, 1)] == 10155547

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["month", "corporate_and_ip_mln_rub", "individual_mln_rub",
                         "total_mln_rub", "source_method"])
        for month in months:
            if month in old_corp:
                corp, ind = old_corp[month][0], old_ind[month][0]
                method = "archive_302_01M_302_02M_cumulative_difference_unverified"
            else:
                corp, ind = new_corp[month], new_ind[month]
                method = "current_monthly_0409303_0409316"
            writer.writerow([month.isoformat(), corp, ind, corp + ind, method])
    print(f"Created {OUTPUT}: {len(old_corp) + len(new_corp)} months")


if __name__ == "__main__":
    main()
