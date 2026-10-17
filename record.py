"""Record raw Binance spot order book diffs, trades and periodic snapshots.

Install: pip install websockets
Run:     caffeinate -i python record.py     (caffeinate keeps your Mac awake)
Stop:    Ctrl+C
Output:  data/raw/<symbol>_<UTC time>.jsonl.gz, one new file per UTC hour
"""
import asyncio
import gzip
import json
import os
import time
import urllib.request

import websockets

SYMBOL = "btcusdt"
STREAMS = f"{SYMBOL}@depth@100ms/{SYMBOL}@trade"
WS_URL = f"wss://data-stream.binance.vision/stream?streams={STREAMS}"
SNAPSHOT_URL = f"https://data-api.binance.vision/api/v3/depth?symbol={SYMBOL.upper()}&limit=1000"
SNAPSHOT_EVERY_S = 600  # a fresh snapshot every 10 minutes lets the rebuild resync after gaps
OUT_DIR = "data/raw"


class HourlyWriter:
    """Writes one JSON line per message into a new gzip file each UTC hour."""

    def __init__(self):
        self.hour, self.f, self.count = None, None, 0

    def write(self, kind, data):
        hour = time.strftime("%Y%m%d_%H", time.gmtime())
        if hour != self.hour:
            self.close()
            os.makedirs(OUT_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            self.f = gzip.open(f"{OUT_DIR}/{SYMBOL}_{stamp}.jsonl.gz", "wt")
            self.hour = hour
        record = {"recv_ns": time.time_ns(), "type": kind, "data": data}
        self.f.write(json.dumps(record) + "\n")
        self.count += 1
        if self.count % 10_000 == 0:
            print(f"{time.strftime('%H:%M:%S')}  {self.count:,} messages recorded")

    def close(self):
        if self.f:
            self.f.close()
            self.f = None


def fetch_snapshot():
    with urllib.request.urlopen(SNAPSHOT_URL, timeout=10) as r:
        return json.load(r)


async def take_snapshot(writer):
    try:
        writer.write("snapshot", await asyncio.to_thread(fetch_snapshot))
    except Exception as e:
        print("snapshot failed:", e)


async def periodic_snapshots(writer):
    while True:
        await asyncio.sleep(SNAPSHOT_EVERY_S)
        await take_snapshot(writer)


async def stream(writer):
    backoff = 1
    tasks = set()
    while True:
        try:
            async with websockets.connect(WS_URL, max_size=2**23) as ws:
                print("connected")
                backoff = 1
                first = True
                async for raw in ws:
                    data = json.loads(raw)["data"]
                    kind = "depth" if data["e"] == "depthUpdate" else "trade"
                    writer.write(kind, data)
                    if first:
                        # Snapshot only after the stream is flowing, so the diffs that
                        # follow the snapshot are guaranteed to be on disk.
                        t = asyncio.create_task(take_snapshot(writer))
                        tasks.add(t)
                        t.add_done_callback(tasks.discard)
                        first = False
        except Exception as e:
            print(f"disconnected ({e!r})")
        print(f"reconnecting in {backoff}s")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60)


async def main():
    writer = HourlyWriter()
    try:
        await asyncio.gather(stream(writer), periodic_snapshots(writer))
    finally:
        writer.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("stopped")
