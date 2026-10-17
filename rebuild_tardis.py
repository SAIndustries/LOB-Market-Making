"""Rebuild the order book from Tardis.dev CSV files into the same Parquet format as
rebuild_book.py (standalone: no other files needed), so 01_explore.ipynb and later steps work unchanged.

Install: pip install sortedcontainers pandas pyarrow
Run:     python rebuild_tardis.py                                   (BTCUSDT -> data/processed)
         python rebuild_tardis.py --symbol DOGEUSDT --out data/processed_doge
Output:  data/processed/book and data/processed/trades (replaces the live-recording output;
         run rebuild_book.py again whenever you want that back)
"""
import argparse
import glob
import os

import pandas as pd
from sortedcontainers import SortedDict

TARDIS_DIR = "data/tardis"
SYMBOL = "BTCUSDT"
OUT_DIR = "data/processed"
LEVELS = 10
CSV_CHUNK = 2_000_000   # rows read at a time, to keep memory use low
CHUNK_ROWS = 200_000    # rows per output Parquet file


class ChunkedParquet:
    def __init__(self, name):
        self.dir = os.path.join(OUT_DIR, name)
        os.makedirs(self.dir, exist_ok=True)
        for old in glob.glob(os.path.join(self.dir, "*.parquet")):
            os.remove(old)
        self.rows, self.part = [], 0

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= CHUNK_ROWS:
            self.flush()

    def flush(self):
        if self.rows:
            pd.DataFrame(self.rows).to_parquet(f"{self.dir}/part_{self.part:04d}.parquet")
            self.rows, self.part = [], self.part + 1


def top(bids, asks, n):
    row = {}
    for i in range(n):
        bp, bq = bids.peekitem(-1 - i) if i < len(bids) else (None, None)
        ap, aq = asks.peekitem(i) if i < len(asks) else (None, None)
        row[f"bid_px_{i}"], row[f"bid_qty_{i}"] = bp, bq
        row[f"ask_px_{i}"], row[f"ask_qty_{i}"] = ap, aq
    return row


def rebuild_book(book_out, stats):
    for path in sorted(glob.glob(f"{TARDIS_DIR}/*incremental_book_L2*_{SYMBOL}.csv.gz")):
        print("book:", path)
        bids, asks = SortedDict(), SortedDict()
        prev_snapshot = False
        msg_local = msg_exch = None  # one exchange message = rows sharing a local_timestamp

        def emit():
            if msg_local is None or not bids or not asks:
                return
            row = {"event_ms": msg_exch // 1000, "recv_ns": msg_local * 1000, **top(bids, asks, LEVELS)}
            if row["bid_px_0"] >= row["ask_px_0"]:
                stats["crossed"] += 1
            book_out.add(row)
            stats["applied"] += 1

        cols = ["timestamp", "local_timestamp", "is_snapshot", "side", "price", "amount"]
        for chunk in pd.read_csv(path, usecols=cols, chunksize=CSV_CHUNK):
            for ts, local, is_snap, side, price, amount in chunk.itertuples(index=False):
                if local != msg_local:
                    emit()
                    msg_local, msg_exch = local, ts
                if is_snap and not prev_snapshot:
                    bids.clear(); asks.clear()     # a new full snapshot starts (e.g. after a reconnect)
                    stats["snapshots"] += 1
                prev_snapshot = bool(is_snap)
                book = bids if side == "bid" else asks
                if amount == 0:
                    book.pop(price, None)
                else:
                    book[price] = amount
        emit()


def rebuild_trades(trade_out, stats):
    for path in sorted(glob.glob(f"{TARDIS_DIR}/*trades*_{SYMBOL}.csv.gz")):
        print("trades:", path)
        cols = ["timestamp", "local_timestamp", "side", "price", "amount"]
        for chunk in pd.read_csv(path, usecols=cols, chunksize=CSV_CHUNK):
            df = pd.DataFrame({
                "trade_ms": chunk["timestamp"] // 1000,
                "recv_ns": chunk["local_timestamp"] * 1000,
                "price": chunk["price"],
                "qty": chunk["amount"],
                "buyer_is_maker": chunk["side"] == "sell",   # Tardis 'side' is the aggressor's side
            })
            for row in df.to_dict("records"):
                trade_out.add(row)
            stats["trades"] += len(df)


def main():
    global SYMBOL, OUT_DIR
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--out", default="data/processed")
    a = p.parse_args()
    SYMBOL, OUT_DIR = a.symbol.upper(), a.out
    stats = {"applied": 0, "snapshots": 0, "crossed": 0, "trades": 0}
    book_out, trade_out = ChunkedParquet("book"), ChunkedParquet("trades")
    rebuild_book(book_out, stats)
    rebuild_trades(trade_out, stats)
    book_out.flush()
    trade_out.flush()
    print(stats)


if __name__ == "__main__":
    main()
