"""Quoting strategies. Each has quote(state) -> (bid_price, ask_price) in ticks, or None
for a side you don't want to quote. state has t (ms), bid_px, bid_qty, ask_px, ask_qty
(lists, best level first) and position.

Strategies:
  SymmetricQuoter     baseline: join the best bid and ask
  AvellanedaStoikov   quotes around an inventory-adjusted reservation price
  ImbalanceFilter     wraps any strategy; pulls the side that order book imbalance
                      says is about to be run over
"""
import math


def imbalance(s):
    b, a = s.bid_qty[0], s.ask_qty[0]
    return (b - a) / (b + a) if b + a > 0 else 0.0


def apply_inventory_limit(bid, ask, position, max_position):
    if position >= max_position:
        bid = None
    if position <= -max_position:
        ask = None
    return bid, ask


class SymmetricQuoter:
    """Baseline: join the best bid and best ask; stop quoting a side at the inventory limit."""

    def __init__(self, max_position):
        self.max_position = max_position

    def quote(self, s):
        return apply_inventory_limit(s.bid_px[0], s.ask_px[0], s.position, self.max_position)


class EwmaVolatility:
    """Variance of 1-second mid-price changes (ticks^2 per second), exponentially weighted."""

    def __init__(self, halflife_s=60, warmup_s=60):
        self.decay = 0.5 ** (1 / halflife_s)
        self.warmup_s = warmup_s
        self.var = None
        self.last_t = self.last_mid = None
        self.samples = 0

    def update(self, t, mid):
        if self.last_t is None:
            self.last_t, self.last_mid = t, mid
            return
        elapsed = t - self.last_t
        if elapsed < 1000:
            return
        r2 = (mid - self.last_mid) ** 2 / (elapsed / 1000)   # scale to a 1-second variance
        self.var = r2 if self.var is None else self.decay * self.var + (1 - self.decay) * r2
        self.last_t, self.last_mid = t, mid
        self.samples += 1

    @property
    def ready(self):
        return self.samples >= self.warmup_s


class AvellanedaStoikov:
    """Avellaneda & Stoikov (2008), with a fixed horizon tau instead of a closing time.

    reservation price  r = mid - q * gamma * sigma^2 * tau
    total spread       d = gamma * sigma^2 * tau + (2 / gamma) * ln(1 + gamma / k)

    q is inventory in units of order_size, sigma^2 is in ticks^2 per second, gamma is risk
    aversion, k is how quickly fill probability falls with distance from the mid (per tick).
    Quotes are never more aggressive than the current best bid / ask (post-only).
    """

    def __init__(self, order_size, max_position, gamma=5e-5, k=1.0, tau_s=1.0, halflife_s=60):
        self.order_size, self.max_position = order_size, max_position
        self.gamma, self.k, self.tau = gamma, k, tau_s
        self.vol = EwmaVolatility(halflife_s)

    def quote(self, s):
        bb, ba = s.bid_px[0], s.ask_px[0]
        mid = (bb + ba) / 2
        self.vol.update(s.t, mid)
        if not self.vol.ready:
            return None, None
        q = s.position / self.order_size
        risk = self.gamma * self.vol.var * self.tau
        r = mid - q * risk
        half = 0.5 * (risk + (2 / self.gamma) * math.log(1 + self.gamma / self.k))
        bid = min(math.floor(r - half), bb)
        ask = max(math.ceil(r + half), ba)
        return apply_inventory_limit(bid, ask, s.position, self.max_position)


class ImbalanceFilter:
    """Wrap a strategy and pull the bid when imbalance < -theta (price likely to fall)
    and the ask when imbalance > theta (price likely to rise)."""

    def __init__(self, inner, theta=0.5):
        self.inner, self.theta = inner, theta

    def quote(self, s):
        bid, ask = self.inner.quote(s)
        imb = imbalance(s)
        if imb < -self.theta:
            bid = None
        if imb > self.theta:
            ask = None
        return bid, ask


def make_strategy(name, order_size, max_position, theta=0.5, gamma=5e-5, k=1.0, tau_s=1.0):
    if name == "symmetric":
        return SymmetricQuoter(max_position)
    if name == "imbalance":
        return ImbalanceFilter(SymmetricQuoter(max_position), theta)
    if name == "as":
        return AvellanedaStoikov(order_size, max_position, gamma, k, tau_s)
    if name == "as_imbalance":
        return ImbalanceFilter(AvellanedaStoikov(order_size, max_position, gamma, k, tau_s), theta)
    raise ValueError(f"unknown strategy {name}")
