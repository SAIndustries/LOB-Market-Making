"""Download free Tardis.dev historical data: the first day of each month needs no API key.

Install: pip install --upgrade tardis-dev   (version 3 or newer)
Run:     python download_tardis.py
Output:  data/tardis/binance_<data type>_<date>_<SYMBOL>.csv.gz
"""
from datetime import date, timedelta

from tardis_dev import download_datasets

SYMBOL = "BTCUSDT"
# First day of each month only (free). Add or remove dates as you like.
DATES = ["2026-08-01", "2026-09-01", "2026-10-01"]

for day in DATES:
    next_day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    print("downloading", day)
    download_datasets(
        exchange="binance",                               # Binance spot
        data_types=["incremental_book_L2", "trades"],
        from_date=day,
        to_date=next_day,                                 # end date is not included
        symbols=[SYMBOL],
        download_dir="data/tardis",
    )
print("done")