"""Unit tests for the fill logic. Run: python -m pytest -q"""
import numpy as np
import pytest

from backtester import BUY, SELL, Backtester, Config

L = 3


def make_book(rows):
    """rows: (time, [(bid_px, qty), ...], [(ask_px, qty), ...]) with prices in ticks."""
    n = len(rows)
    b = {"time": np.array([r[0] for r in rows], np.int64),
         "bid_px": -np.ones((n, L), np.int64), "bid_qty": np.zeros((n, L)),
         "ask_px": -np.ones((n, L), np.int64), "ask_qty": np.zeros((n, L))}
    for i, (_, bids, asks) in enumerate(rows):
        for k, (p, q) in enumerate(bids):
            b["bid_px"][i, k], b["bid_qty"][i, k] = p, q
        for k, (p, q) in enumerate(asks):
            b["ask_px"][i, k], b["ask_qty"][i, k] = p, q
    return b


def make_trades(rows):
    """rows: (time, price_ticks, qty, aggressor) with aggressor BUY or SELL."""
    return {"time": np.array([r[0] for r in rows], np.int64), "price": np.array([r[1] for r in rows], np.int64),
            "qty": np.array([r[2] for r in rows], float), "aggressor": np.array([r[3] for r in rows], np.int64)}


class Schedule:
    """Strategy that returns whatever quotes fn(t) gives."""
    def __init__(self, fn):
        self.fn = fn

    def quote(self, s):
        return self.fn(s.t)


BOOK0 = (0, [(99, 5)], [(101, 5)])
CFG = Config(tick=1.0, order_size=1.0, latency_ms=10)


def run(book_rows, trade_rows, fn, **cfg):
    c = Config(**{**CFG.__dict__, **cfg})
    return Backtester(Schedule(fn), c).run(make_book(book_rows), make_trades(trade_rows))


def test_queue_ahead_is_eaten_before_we_fill():
    res = run([BOOK0], [(20, 99, 3, SELL), (30, 99, 3, SELL)], lambda t: (99, None))
    f = res["fills"]
    assert len(f) == 1 and f.time[0] == 30 and f.qty[0] == 1     # 5 ahead: 3 + 2 eaten, 1 left for us


def test_no_fill_before_order_is_live():
    res = run([BOOK0], [(5, 98, 10, SELL), (20, 98, 10, SELL)], lambda t: (99, None))
    assert list(res["fills"].time) == [20]


def test_post_only_order_is_rejected():
    res = run([BOOK0], [(20, 101, 1, BUY)], lambda t: (101, None))
    assert res["n_rejected"] == 1 and len(res["fills"]) == 0


def test_can_fill_while_cancel_is_in_flight():
    books = [BOOK0, (50, [(99, 5)], [(101, 5)])]
    res = run(books, [(55, 98, 10, SELL)], lambda t: (99, None) if t < 50 else (None, None))
    assert len(res["fills"]) == 1                              # cancel at 50 only takes effect at 60


def test_no_fill_after_cancel_takes_effect():
    books = [BOOK0, (50, [(99, 5)], [(101, 5)])]
    res = run(books, [(65, 98, 10, SELL)], lambda t: (99, None) if t < 50 else (None, None))
    assert len(res["fills"]) == 0


def test_book_moving_through_our_price_fills_us():
    books = [BOOK0, (20, [(99, 5)], [(101, 5)]), (30, [(97, 5)], [(99, 2)])]
    res = run(books, [], lambda t: (99, None))
    assert len(res["fills"]) == 1 and res["fills"].time[0] == 30


@pytest.mark.parametrize("model, expected_fills", [("pessimistic", 0), ("proportional", 1)])
def test_queue_models(model, expected_fills):
    books = [(0, [(99, 10)], [(101, 5)]),
             (20, [(99, 10)], [(101, 5)]),   # we go live at 10 with 10 ahead
             (30, [(99, 20)], [(101, 5)]),   # 10 join behind us
             (40, [(99, 10)], [(101, 5)])]   # 10 cancel
    res = run(books, [(50, 99, 6, SELL)], lambda t: (99, None), queue_model=model)
    # pessimistic: cancels were behind us, 10 still ahead -> no fill
    # proportional: half the cancels were ahead -> 5 ahead -> 6 traded fills our 1
    assert len(res["fills"]) == expected_fills


def test_pnl_accounting():
    books = [BOOK0]
    trades = [(20, 99, 10, SELL), (30, 101, 10, BUY)]
    res = run(books, trades, lambda t: (99, 101))
    eq = res["equity"].iloc[-1]
    assert eq.position == 0
    assert eq.gross_pnl == pytest.approx(2.0)                  # bought 99, sold 101
    assert eq.notional == pytest.approx(200.0)
