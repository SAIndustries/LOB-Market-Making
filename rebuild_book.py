"""Rebuild the order book from raw recordings; save top-of-book and trades as Parquet.

Install: pip install sortedcontainers pandas pyarrow
Run:     python rebuild_book.py
Output:  data/processed/book/part_*.parquet    top LEVELS of the book after every update
         data/processed/trades/part_*.parquet  every trade
Load:    pd.read_parquet("data/processed/book")
"""
import glob
import gzip
import json
import os
from collections import deque

import pandas as pd
from sortedcontainers import SortedDict

RAW_GLOB = "data/raw/*.jsonl.gz"
OUT_DIR = "data/processed"
LEVELS = 10
CHUNK_ROWS = 200_000  # write to disk in chunks so a day of data fits in memory


def read_records():
    for path in sorted(glob.glob(RAW_GLOB)):
        with gzip.open(path, "rt") as f:
            try:
                for line in f:
                    yield json.loads(line)
            except (EOFError, OSError, json.JSONDecodeError):
                print(f"warning: {path} ends early (recorder was stopped); used what was readable")


class Book:
    def __init__(self):
        self.bids = SortedDict()  # price -> qty; best bid is the last key
        self.asks = SortedDict()  # best ask is the first key
        self.last_id = None       # last update id applied; None means not synced
        self.awaiting_first = False

    def load_snapshot(self, snap):
        self.bids = SortedDict({float(p): float(q) for p, q in snap["bids"]})
        self.asks = SortedDict({float(p): float(q) for p, q in snap["asks"]})
        self.last_id = snap["lastUpdateId"]
        self.awaiting_first = True

    def apply_diff(self, d):
        """Apply one diff. Returns 'applied', 'skipped' or 'gap'."""
        if self.last_id is None:
            return "skipped"                       # waiting for a snapshot
        first_id, last_id = d["U"], d["u"]
        if last_id <= self.last_id:
            return "skipped"                       # already included in the snapshot
        if self.awaiting_first:
            if not (first_id <= self.last_id + 1 <= last_id):
                self.last_id = None                # snapshot doesn't line up; wait for the next
                return "gap"
            self.awaiting_first = False
        elif first_id != self.last_id + 1:
            self.last_id = None                    # missed messages; resync from next snapshot
            return "gap"
        for side, key in ((self.bids, "b"), (self.asks, "a")):
            for p, q in d[key]:
                p, q = float(p), float(q)
                if q == 0:
                    side.pop(p, None)
                else:
                    side[p] = q
        self.last_id = last_id
        return "applied"

    def top(self, n):
        row = {}
        for i in range(n):
            bp, bq = self.bids.peekitem(-1 - i) if i < len(self.bids) else (None, None)
            ap, aq = self.asks.peekitem(i) if i < len(self.asks) else (None, None)
            row[f"bid_px_{i}"], row[f"bid_qty_{i}"] = bp, bq
            row[f"ask_px_{i}"], row[f"ask_qty_{i}"] = ap, aq
        return row


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


def main():
    book = Book()
    recent = deque(maxlen=2000)  # diffs that arrived before their snapshot did
    book_out, trade_out = ChunkedParquet("book"), ChunkedParquet("trades")
    stats = {"applied": 0, "skipped": 0, "gap": 0, "crossed": 0, "trades": 0}

    def handle(d, recv_ns):
        result = book.apply_diff(d)
        stats[result] += 1
        if result == "applied":
            row = {"event_ms": d["E"], "recv_ns": recv_ns, **book.top(LEVELS)}
            if row["bid_px_0"] is not None and row["ask_px_0"] is not None \
                    and row["bid_px_0"] >= row["ask_px_0"]:
                stats["crossed"] += 1              # should stay at 0 if the rebuild is right
            book_out.add(row)

    for rec in read_records():
        kind, d = rec["type"], rec["data"]
        if kind == "trade":
            stats["trades"] += 1
            trade_out.add({"trade_ms": d["T"], "recv_ns": rec["recv_ns"],
                           "price": float(d["p"]), "qty": float(d["q"]),
                           "buyer_is_maker": d["m"]})
        elif kind == "depth":
            recent.append((d, rec["recv_ns"]))
            handle(d, rec["recv_ns"])
        elif kind == "snapshot" and book.last_id is None:
            book.load_snapshot(d)
            for old, ns in recent:                 # replay diffs buffered before the snapshot
                handle(old, ns)

    book_out.flush()
    trade_out.flush()
    print(stats)


if __name__ == "__main__":
    main()
