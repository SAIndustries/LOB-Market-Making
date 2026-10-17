"""Final runs for the write-up: the configurations chosen on the tuning days, on every day,
for both instruments. Saves fills and equity curves for 03_evaluation.ipynb.

Run:     python run_final.py
Output:  results/final/<SYMBOL>/<strategy>/fills_<day>.parquet, equity_<day>.parquet
         results/final/summary.csv
"""
import os

import pandas as pd

from backtester import Backtester, Config, summarize
from run_backtest import load, to_arrays
from strategies import make_strategy

# Parameters chosen by run_experiments.py on the tuning days (2026-08-01, 2026-09-01).
INSTRUMENTS = {
    "BTCUSDT": dict(data="data/processed", tick=0.01, order_size=0.001, max_position=0.01, strategies=[
        ("symmetric", {}), ("imbalance", {"theta": 0.8}), ("as", {"gamma": 0.005}),
        ("as_imbalance", {"gamma": 0.005, "theta": 0.8})]),
    "DOGEUSDT": dict(data="data/processed_doge", tick=0.00001, order_size=500, max_position=5000, strategies=[
        ("symmetric", {}), ("imbalance", {"theta": 0.6}), ("as", {"gamma": 0.0001}),
        ("as_imbalance", {"gamma": 0.0001, "theta": 0.6})]),
}
LATENCY_MS = 200
QUEUE_MODEL = "proportional"
OUT = "results/final"


def label(name, params):
    return name + "".join(f"_{k}={v:g}" for k, v in params.items())


def main():
    rows = []
    for symbol, spec in INSTRUMENTS.items():
        cfg = Config(tick=spec["tick"], order_size=spec["order_size"], latency_ms=LATENCY_MS, queue_model=QUEUE_MODEL)
        book, trades = load(spec["data"])
        days = sorted(book.day.unique())
        arrays = {d: to_arrays(book[book.day == d], trades[trades.day == d], cfg.tick) for d in days}
        for name, params in spec["strategies"]:
            lab = label(name, params)
            out_dir = os.path.join(OUT, symbol, lab)
            os.makedirs(out_dir, exist_ok=True)
            for day, (b, t) in arrays.items():
                print(f"{symbol} {lab} {day}")
                strat = make_strategy(name, spec["order_size"], spec["max_position"], **params)
                res = Backtester(strat, cfg).run(b, t)
                res["fills"].to_parquet(f"{out_dir}/fills_{day}.parquet")
                res["equity"].to_parquet(f"{out_dir}/equity_{day}.parquet")
                rows.append({"symbol": symbol, "strategy": lab, "day": str(day), **summarize(res, b, cfg)})
    pd.DataFrame(rows).to_csv(f"{OUT}/summary.csv", index=False)
    print("saved", f"{OUT}/summary.csv")


if __name__ == "__main__":
    main()
