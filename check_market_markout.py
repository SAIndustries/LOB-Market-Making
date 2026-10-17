"""Validation: how did ALL passive (maker) orders in the real market do?

For every real trade, the passive side's markout is measured exactly as for our simulated
fills: side * (mid after h seconds - trade price). If the market-wide figure is close to
our backtester's markouts, the simulator is reflecting the real market, not a bug.

Run:  python check_market_markout.py                                        (BTCUSDT)
      python check_market_markout.py --data data/processed_doge --tick 0.00001
"""
import argparse

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/processed")
p.add_argument("--tick", type=float, default=0.01)
a = p.parse_args()

book = pd.read_parquet(f"{a.data}/book", columns=["event_ms", "bid_px_0", "ask_px_0"]).sort_values("event_ms")
trades = pd.read_parquet(f"{a.data}/trades", columns=["trade_ms", "price", "qty", "buyer_is_maker"])
bt = book.event_ms.to_numpy()
mid = ((book.bid_px_0 + book.ask_px_0) / 2).to_numpy()
price = trades.price.to_numpy()
maker = np.where(trades.buyer_is_maker.to_numpy(bool), 1, -1)   # +1 if the passive side bought
day = pd.to_datetime(trades.trade_ms, unit="ms").dt.date.to_numpy()
book_day = pd.to_datetime(book.event_ms, unit="ms").dt.date.to_numpy()

rows = []
for h in (0, 1, 10):
    target = trades.trade_ms.to_numpy() + h * 1000
    idx = np.searchsorted(bt, target, side="right") - 1
    ok = (idx >= 0) & (book_day[np.clip(idx, 0, None)] == day)        # stay within the same day
    if h:
        ok &= target <= bt[-1]
    mk_ticks = maker[ok] * (mid[idx[ok]] - price[ok]) / a.tick
    mk_bps = 1e4 * maker[ok] * (mid[idx[ok]] - price[ok]) / price[ok]
    w = trades.qty.to_numpy()[ok] * price[ok]                         # weight by traded value
    rows.append({"horizon_s": h, "trades": int(ok.sum()),
                 "markout_ticks": mk_ticks.mean(), "markout_bps_value_weighted": np.average(mk_bps, weights=w)})

print(pd.DataFrame(rows).set_index("horizon_s").round(3).to_string())
