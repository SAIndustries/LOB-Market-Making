"""Unit tests for the strategies. Run: python -m pytest -q"""
from types import SimpleNamespace

from strategies import AvellanedaStoikov, ImbalanceFilter, SymmetricQuoter


def state(t=0, bid=(100, 1.0), ask=(101, 1.0), position=0.0):
    return SimpleNamespace(t=t, bid_px=[bid[0]], bid_qty=[bid[1]], ask_px=[ask[0]], ask_qty=[ask[1]],
                           position=position)


def test_symmetric_joins_best_and_respects_limit():
    s = SymmetricQuoter(max_position=0.01)
    assert s.quote(state()) == (100, 101)
    assert s.quote(state(position=0.01)) == (None, 101)
    assert s.quote(state(position=-0.01)) == (100, None)


def test_imbalance_filter_pulls_the_threatened_side():
    f = ImbalanceFilter(SymmetricQuoter(1.0), theta=0.5)
    assert f.quote(state(bid=(100, 0.1), ask=(101, 1.0))) == (None, 101)   # ask-heavy: price likely down
    assert f.quote(state(bid=(100, 1.0), ask=(101, 0.1))) == (100, None)   # bid-heavy: price likely up
    assert f.quote(state(bid=(100, 1.0), ask=(101, 0.9))) == (100, 101)


def warmed_up(strategy, seconds=120):
    for i in range(seconds):
        mid_shift = 10 if i % 2 else 0                 # mid moves 10 ticks every second
        strategy.quote(state(t=i * 1000, bid=(100 + mid_shift, 1), ask=(101 + mid_shift, 1)))
    return strategy


def test_as_never_quotes_through_the_best_prices():
    s = warmed_up(AvellanedaStoikov(0.001, 1.0, gamma=1e-3, k=1.0))
    bid, ask = s.quote(state(t=200_000))
    assert bid <= 100 and ask >= 101


def test_as_skews_quotes_down_when_long():
    s = warmed_up(AvellanedaStoikov(0.001, 1.0, gamma=0.05, k=1.0))
    book = dict(t=200_000, bid=(124, 1), ask=(126, 1))
    flat_bid, _ = s.quote(state(**book))
    long_bid, long_ask = s.quote(state(**book, position=0.005))
    assert long_bid < flat_bid                 # long: bid further away, less likely to buy more
    assert long_ask == 126                     # and the ask moves to the best ask to sell sooner


def test_as_waits_for_volatility_estimate():
    s = AvellanedaStoikov(0.001, 1.0)
    assert s.quote(state(t=0)) == (None, None)
