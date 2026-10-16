"""Compare strategies with an honest tune/test split.

1. Tune: run every configuration on the tuning day(s) and pick the best of each family.
2. Combine: try Avellaneda-Stoikov + imbalance filter using the best parameters found.
3. Test: run only the chosen configurations on the test day, which was never used to choose.

Run:     python run_experiments.py
         python run_experiments.py --tune-days 2026-08-01 2026-09-01 --test-days 2026-10-01
Output:  results/experiments/tune.csv, test.csv and fills for the test runs
"""
import argparse
import os

import pandas as pd

from backtester import Backtester, Config, summarize
from run_backtest import load, to_arrays
from strategies import make_strategy

GRID = (
    [("symmetric", {})]
    + [("imbalance", {"theta": th}) for th in (0.2, 0.4, 0.6, 0.8)]
    + [("as", {"gamma": g}) for g in (2e-5, 5e-5, 1e-4, 2e-4)]
)


def label(name, params):
    return name + "".join(f" {k}={v:g}" for k, v in params.items())


def run_one(arrays, name, params, cfg, a):
    rows, fills = [], {}
    for day, (b, t) in arrays.items():
        strat = make_strategy(name, a.order_size, a.max_position, **params)
        res = Backtester(strat, cfg).run(b, t)
        rows.append(summarize(res, b, cfg, fees_bps=a.fees_bps))
        fills[day] = res["fills"]
    df = pd.DataFrame(rows)
    out = {"strategy": label(name, params), "fills": df.fills.sum(), "volume": df.volume.sum(),
           "gross_pnl": df.gross_pnl.sum()}
    out["pnl_per_btc"] = out["gross_pnl"] / out["volume"] if out["volume"] else float("nan")
    for f in a.fees_bps:
        out[f"net_{f:g}bps"] = df[f"net_pnl_fee_{f}bps"].sum()
    w = df.fills / df.fills.sum() if df.fills.sum() else df.fills
    for col in ("spread_capture_ticks", "markout_1s_ticks", "markout_10s_ticks"):
        out[col] = (df[col].fillna(0) * w).sum()            # fill-weighted average across days
    out["max_abs_position"] = df.max_abs_position.max()
    return out, fills


def best(table, family):
    rows = table[table.strategy.str.startswith(family + " ") | (table.strategy == family)]
    return rows.sort_values("gross_pnl", ascending=False).iloc[0]


def parse_params(lbl):
    parts = lbl.split()
    return parts[0], {k: float(v) for k, v in (p.split("=") for p in parts[1:])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--tune-days", nargs="+", default=["2026-08-01", "2026-09-01"])
    p.add_argument("--test-days", nargs="+", default=["2026-10-01"])
    p.add_argument("--tick", type=float, default=0.01)
    p.add_argument("--order-size", type=float, default=0.001)
    p.add_argument("--max-position", type=float, default=0.01)
    p.add_argument("--latency-ms", type=int, default=200)
    p.add_argument("--queue-model", default="proportional", choices=["proportional", "pessimistic"])
    p.add_argument("--fees-bps", type=float, nargs="+", default=[10, 2, 0, -0.5])
    a = p.parse_args()

    cfg = Config(tick=a.tick, order_size=a.order_size, latency_ms=a.latency_ms, queue_model=a.queue_model)
    book, trades = load(a.data)

    def arrays_for(days):
        out = {}
        for d in days:
            d = pd.Timestamp(d).date()
            out[d] = to_arrays(book[book.day == d], trades[trades.day == d], cfg.tick)
        return out

    out_dir = "results/experiments"
    os.makedirs(out_dir, exist_ok=True)
    show = lambda df: print(df.set_index("strategy").round(3).to_string(), "\n")

    print("== TUNING on", a.tune_days)
    tune_arrays = arrays_for(a.tune_days)
    rows = []
    for name, params in GRID:
        print("  running", label(name, params))
        rows.append(run_one(tune_arrays, name, params, cfg, a)[0])
    tune = pd.DataFrame(rows)

    best_imb, best_as = best(tune, "imbalance"), best(tune, "as")
    theta = parse_params(best_imb.strategy)[1]["theta"]
    gamma = parse_params(best_as.strategy)[1]["gamma"]
    combo = ("as_imbalance", {"gamma": gamma, "theta": theta})
    print("  running", label(*combo))
    tune = pd.concat([tune, pd.DataFrame([run_one(tune_arrays, *combo, cfg, a)[0]])], ignore_index=True)
    tune.to_csv(f"{out_dir}/tune.csv", index=False)
    show(tune)

    chosen = [("symmetric", {}), parse_params(best_imb.strategy), parse_params(best_as.strategy), combo]
    print("== TEST on", a.test_days, "(configurations chosen on the tuning days only)")
    test_arrays = arrays_for(a.test_days)
    rows = []
    for name, params in chosen:
        print("  running", label(name, params))
        row, fills = run_one(test_arrays, name, params, cfg, a)
        rows.append(row)
        for day, f in fills.items():
            f.to_parquet(f"{out_dir}/test_fills_{label(name, params).replace(' ', '_')}_{day}.parquet")
    test = pd.DataFrame(rows)
    test.to_csv(f"{out_dir}/test.csv", index=False)
    show(test)


if __name__ == "__main__":
    main()
