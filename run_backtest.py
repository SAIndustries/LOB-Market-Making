"""Run a strategy through the backtester, one day at a time.

Run:     python run_backtest.py                      (baseline, default settings)
         python run_backtest.py --latency-ms 50 --queue-model pessimistic --days 2026-10-01
         python run_backtest.py --strategy imbalance --theta 0.6 --name imb_06
         python run_backtest.py --strategy as --gamma 5e-5 --k 1 --name as
Output:  a summary table, plus results/<name>/ with fills, equity curve and summary per day
"""
import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import pandas as pd

from backtester import Backtester, Config, summarize
from strategies import make_strategy

LEVELS = 5


def load(data_dir):
    cols = ["event_ms"] + [f"{s}_{k}_{i}" for i in range(LEVELS) for s in ("bid", "ask") for k in ("px", "qty")]
    book = pd.read_parquet(f"{data_dir}/book", columns=cols).sort_values("event_ms", kind="stable")
    trades = pd.read_parquet(f"{data_dir}/trades", columns=["trade_ms", "price", "qty", "buyer_is_maker"])
    trades = trades.sort_values("trade_ms", kind="stable")
    book["day"] = pd.to_datetime(book.event_ms, unit="ms").dt.date
    trades["day"] = pd.to_datetime(trades.trade_ms, unit="ms").dt.date
    return book, trades


def to_arrays(book, trades, tick):
    def px(side):
        a = book[[f"{side}_px_{i}" for i in range(LEVELS)]].to_numpy(float)
        return np.where(np.isnan(a), -1, np.rint(a / tick)).astype(np.int64)

    def qty(side):
        return book[[f"{side}_qty_{i}" for i in range(LEVELS)]].fillna(0).to_numpy(float)

    b = {"time": book.event_ms.to_numpy(np.int64), "bid_px": px("bid"), "bid_qty": qty("bid"),
         "ask_px": px("ask"), "ask_qty": qty("ask")}
    t = {"time": trades.trade_ms.to_numpy(np.int64),
         "price": np.rint(trades.price.to_numpy(float) / tick).astype(np.int64),
         "qty": trades.qty.to_numpy(float),
         "aggressor": np.where(trades.buyer_is_maker.to_numpy(bool), -1, 1)}  # maker buyer => seller hit
    return b, t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--name", default="baseline")
    p.add_argument("--days", nargs="*", help="e.g. 2026-10-01 (default: all days)")
    p.add_argument("--tick", type=float, default=0.01)
    p.add_argument("--order-size", type=float, default=0.001)
    p.add_argument("--max-position", type=float, default=0.01)
    p.add_argument("--latency-ms", type=int, default=200)
    p.add_argument("--queue-model", default="proportional", choices=["proportional", "pessimistic"])
    p.add_argument("--fees-bps", type=float, nargs="+", default=[10, 2, 0, -0.5])
    p.add_argument("--strategy", default="symmetric", choices=["symmetric", "imbalance", "as", "as_imbalance"])
    p.add_argument("--theta", type=float, default=0.5, help="imbalance threshold")
    p.add_argument("--gamma", type=float, default=5e-5, help="Avellaneda-Stoikov risk aversion")
    p.add_argument("--k", type=float, default=1.0, help="Avellaneda-Stoikov fill-intensity decay")
    p.add_argument("--tau", type=float, default=1.0, help="Avellaneda-Stoikov horizon (seconds)")
    a = p.parse_args()

    cfg = Config(tick=a.tick, order_size=a.order_size, latency_ms=a.latency_ms, queue_model=a.queue_model)
    book, trades = load(a.data)
    days = [pd.Timestamp(d).date() for d in a.days] if a.days else sorted(book.day.unique())
    out_dir = os.path.join("results", a.name)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for day in days:
        print(f"{day}: running")
        b, t = to_arrays(book[book.day == day], trades[trades.day == day], cfg.tick)
        strat = make_strategy(a.strategy, a.order_size, a.max_position, a.theta, a.gamma, a.k, a.tau)
        res = Backtester(strat, cfg).run(b, t, progress=True)
        summary = {"day": str(day), **summarize(res, b, cfg, fees_bps=a.fees_bps)}
        rows.append(summary)
        res["fills"].to_parquet(f"{out_dir}/fills_{day}.parquet")
        res["equity"].to_parquet(f"{out_dir}/equity_{day}.parquet")

    table = pd.DataFrame(rows).set_index("day")
    table.to_csv(f"{out_dir}/summary.csv")
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump({**asdict(cfg), **{k: v for k, v in vars(a).items() if k not in ("data", "days")}}, f, indent=2)
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:,.4f}".format):
        print(table.T)


if __name__ == "__main__":
    main()
