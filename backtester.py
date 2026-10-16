"""Event-driven backtester for passive (maker) quoting on L2 order book data.

Replays book updates and trades in exchange-time order and simulates our own limit
orders with:
  - latency: an order or cancel decided at time t takes effect at t + latency_ms
  - queue position: when an order goes live it joins the back of the displayed queue
    at its price; trades at that price eat the queue ahead of it first
  - cancellations by others: "pessimistic" assumes they all come from behind us,
    "proportional" spreads them evenly through the queue
  - partial fills, post-only rejection (an order that would cross is rejected)
  - fills when a trade or the book moves through our price

Assumption: our orders are small enough not to change what other traders do.
All prices are integer ticks inside the engine. PnL is tracked before fees; fees are
applied in summarize() so you can compare fee levels from a single run.
"""
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

BUY, SELL = 1, -1


@dataclass
class Config:
    tick: float = 0.01               # price of one tick
    order_size: float = 0.001        # quantity per order (base currency, e.g. BTC)
    latency_ms: int = 200            # decision -> order/cancel active at the exchange
    queue_model: str = "proportional"  # or "pessimistic"
    record_every_ms: int = 1000      # how often to record the equity curve


class Order:
    __slots__ = ("id", "side", "price", "qty", "live_at", "cancel_at", "live", "done",
                 "ahead", "level_qty", "traded")

    def __init__(self, oid, side, price, qty, live_at):
        self.id, self.side, self.price, self.qty = oid, side, price, qty
        self.live_at, self.cancel_at = live_at, math.inf
        self.live = self.done = False
        self.ahead = self.level_qty = self.traded = 0.0


class MarketState:
    """What the strategy sees at each book update (prices in ticks)."""
    __slots__ = ("t", "bid_px", "bid_qty", "ask_px", "ask_qty", "position")


class Backtester:
    def __init__(self, strategy, config=None):
        self.cfg = config or Config()
        if self.cfg.queue_model not in ("proportional", "pessimistic"):
            raise ValueError("queue_model must be 'proportional' or 'pessimistic'")
        self.strategy = strategy
        self.orders = []
        self.next_id = 0
        self.position = 0.0
        self.cash = 0.0       # in ticks x quantity, before fees
        self.notional = 0.0   # traded value in ticks x quantity (fees are a % of this)
        self.fills = []
        self.equity = []
        self.n_placed = self.n_rejected = 0
        self.last_record = -math.inf
        self.bpx = self.bqty = self.apx = self.aqty = None
        self.state = MarketState()

    # ---------- main loop ----------
    def run(self, book, trades, progress=False):
        bt, bpx, bq, apx, aq = book["time"], book["bid_px"], book["bid_qty"], book["ask_px"], book["ask_qty"]
        tt, tpx, tq, tag = trades["time"], trades["price"], trades["qty"], trades["aggressor"]
        nb, nt = len(bt), len(tt)
        i = j = 0
        total, step = nb + nt, max((nb + nt) // 10, 1)
        while i < nb or j < nt:
            # Trades stamped at or before a book update happened before it.
            if j < nt and (i >= nb or tt[j] <= bt[i]):
                t = int(tt[j])
                self._advance(t)
                if self.bpx is not None:
                    self._on_trade(t, int(tpx[j]), float(tq[j]), int(tag[j]))
                j += 1
            else:
                t = int(bt[i])
                self._advance(t)
                self.bpx, self.bqty = bpx[i].tolist(), bq[i].tolist()
                self.apx, self.aqty = apx[i].tolist(), aq[i].tolist()
                self._on_book(t)
                i += 1
            if progress and (i + j) % step == 0:
                print(f"  {100 * (i + j) // total}%")
        self._record(t if total else 0, force=True)
        return self._result()

    # ---------- order lifecycle ----------
    def _advance(self, t):
        """Apply order activations and cancels that have become effective by time t."""
        events = []
        for o in self.orders:
            if o.done:
                continue
            if not o.live and o.live_at <= t and o.live_at < o.cancel_at:
                events.append((o.live_at, 0, o))
            if o.cancel_at <= t:
                events.append((o.cancel_at, 1, o))
        for _, kind, o in sorted(events, key=lambda e: (e[0], e[1])):
            if o.done:
                continue
            if kind == 0:
                self._activate(o)
            else:
                o.done, o.live = True, False

    def _activate(self, o):
        bb, ba = self._best()
        if (o.side == BUY and ba is not None and o.price >= ba) or \
           (o.side == SELL and bb is not None and o.price <= bb):
            o.done = True              # post-only: would have crossed the spread
            self.n_rejected += 1
            return
        qty, known = self._displayed(o.side, o.price)
        o.live = True
        o.ahead = qty if known else math.inf   # unknown queue: assume we are far back
        o.level_qty, o.traded = qty, 0.0

    def _place(self, t, side, price):
        o = Order(self.next_id, side, price, self.cfg.order_size, t + self.cfg.latency_ms)
        self.next_id += 1
        self.n_placed += 1
        self.orders.append(o)

    # ---------- market events ----------
    def _on_trade(self, t, price, qty, aggressor):
        for o in self.orders:
            if not o.live or o.side != -aggressor:
                continue
            through = price < o.price if o.side == BUY else price > o.price
            if through:
                self._fill(t, o, o.qty, "trade_through")
            elif price == o.price:
                o.traded += qty
                eaten = min(o.ahead, qty)
                o.ahead -= eaten
                left = qty - eaten
                if left > 0:
                    self._fill(t, o, min(left, o.qty), "queue")

    def _on_book(self, t):
        bb, ba = self._best()
        for o in self.orders:
            if not o.live:
                continue
            if (o.side == BUY and ba is not None and ba <= o.price) or \
               (o.side == SELL and bb is not None and bb >= o.price):
                self._fill(t, o, o.qty, "book_through")   # the market moved through our price
                continue
            qty, known = self._displayed(o.side, o.price)
            if not known:
                continue
            if self.cfg.queue_model == "proportional" and o.ahead != math.inf:
                remaining = o.level_qty - o.traded
                cancels = remaining - qty
                if cancels > 0 and remaining > 0:
                    o.ahead -= cancels * o.ahead / remaining
            o.ahead = min(o.ahead, qty)
            o.level_qty, o.traded = qty, 0.0

        if bb is None or ba is None:
            return
        self._record(t)

        s = self.state
        s.t, s.position = t, self.position
        s.bid_px, s.bid_qty, s.ask_px, s.ask_qty = self.bpx, self.bqty, self.apx, self.aqty
        want = dict(zip((BUY, SELL), self.strategy.quote(s)))
        for side, price in want.items():
            keep = False
            for o in self.orders:
                if o.done or o.side != side or o.cancel_at != math.inf:
                    continue
                if price is not None and o.price == price and not keep:
                    keep = True                   # same price: keep our queue position
                else:
                    o.cancel_at = t + self.cfg.latency_ms
            if price is not None and not keep:
                self._place(t, side, int(price))
        self.orders = [o for o in self.orders if not o.done]

    def _fill(self, t, o, qty, reason):
        if qty <= 0:
            return
        bb, ba = self._best()
        mid = (bb + ba) / 2 if bb is not None and ba is not None else float(o.price)
        self.position += o.side * qty
        self.cash -= o.side * qty * o.price
        self.notional += qty * o.price
        self.fills.append((t, o.side, o.price, qty, mid, o.id, reason))
        o.qty -= qty
        if o.qty <= 1e-12:
            o.done, o.live = True, False

    # ---------- helpers ----------
    def _best(self):
        bb = self.bpx[0] if self.bpx and self.bpx[0] > 0 else None
        ba = self.apx[0] if self.apx and self.apx[0] > 0 else None
        return bb, ba

    def _displayed(self, side, price):
        """Displayed size at a price, and whether we can know it from the visible levels."""
        px, qt = (self.bpx, self.bqty) if side == BUY else (self.apx, self.aqty)
        valid = [p for p in px if p > 0]
        for p, q in zip(px, qt):
            if p == price:
                return q, True
        if not valid:
            return 0.0, False
        known = price >= min(valid) if side == BUY else price <= max(valid)
        return 0.0, known                     # inside the visible range but empty

    def _record(self, t, force=False):
        bb, ba = self._best()
        if bb is None or ba is None:
            return
        if force or t - self.last_record >= self.cfg.record_every_ms:
            self.equity.append((t, (bb + ba) / 2, self.position, self.cash, self.notional))
            self.last_record = t

    def _result(self):
        tick = self.cfg.tick
        fills = pd.DataFrame(self.fills, columns=["time", "side", "price_ticks", "qty", "mid_ticks", "order_id", "reason"])
        eq = pd.DataFrame(self.equity, columns=["time", "mid_ticks", "position", "cash_ticks", "notional_ticks"])
        eq["gross_pnl"] = (eq.cash_ticks + eq.position * eq.mid_ticks) * tick
        eq["notional"] = eq.notional_ticks * tick
        return {"fills": fills, "equity": eq, "n_placed": self.n_placed, "n_rejected": self.n_rejected}


def summarize(result, book, config, fees_bps=(10, 2, 0, -0.5), horizons_ms=(1000, 10000)):
    """Headline numbers for one run. Markouts and spread capture are in ticks per fill."""
    fills, eq, tick = result["fills"], result["equity"], config.tick
    out = {"orders_placed": result["n_placed"], "rejected_post_only": result["n_rejected"],
           "fills": len(fills)}
    if len(eq):
        last = eq.iloc[-1]
        out.update({"volume": fills.qty.sum(), "notional": last.notional,
                    "end_position": last.position, "max_abs_position": eq.position.abs().max(),
                    "gross_pnl": last.gross_pnl})
        for f in fees_bps:
            out[f"net_pnl_fee_{f}bps"] = last.gross_pnl - last.notional * f / 1e4
    if len(fills):
        btime = book["time"]
        bmid = (book["bid_px"][:, 0] + book["ask_px"][:, 0]) / 2
        for r in ("queue", "trade_through", "book_through"):
            out[f"fills_{r}_pct"] = 100 * float((fills.reason == r).mean())
        out["spread_capture_ticks"] = float((fills.side * (fills.mid_ticks - fills.price_ticks)).mean())
        for h in horizons_ms:
            idx = np.searchsorted(btime, fills.time.to_numpy() + h, side="right") - 1
            ok = (fills.time.to_numpy() + h) <= btime[-1]
            mk = fills.side.to_numpy() * (bmid[idx] - fills.price_ticks.to_numpy())
            out[f"markout_{h // 1000}s_ticks"] = float(mk[ok].mean()) if ok.any() else float("nan")
            if h == horizons_ms[0]:
                for r in ("queue", "trade_through", "book_through"):
                    m = ok & (fills.reason.to_numpy() == r)
                    out[f"markout_{h // 1000}s_{r}_ticks"] = float(mk[m].mean()) if m.any() else float("nan")
    return out
