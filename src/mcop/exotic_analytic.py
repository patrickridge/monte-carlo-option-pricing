"""
Closed-form prices for the exotic payoffs that have them.

Every Monte Carlo price needs something to be checked against, and exotics make
that harder than vanillas: most of them have no closed form at all, which is
precisely why you reach for simulation. The ones collected here are the
exceptions, and each earns its place for a specific reason:

- **Digitals** have a clean closed form and a discontinuous payoff, which makes
  them the ideal test case for the Greek estimators. See `mcop.greeks`: the
  pathwise method breaks on them and the likelihood-ratio method does not.
- **Geometric-average Asians** have a closed form because the geometric mean of
  lognormal variables is itself lognormal. The arithmetic average is not, and
  has no exact formula — so the geometric price does double duty as an exact
  validation target *and* as a control variate for pricing the arithmetic
  version (`mcop.exotics.asian_price_with_control`).
- **Down-and-out / down-and-in calls** have a closed form under *continuous*
  monitoring. That is the number a discretely-monitored simulation should
  converge toward as the barrier-crossing correction improves, which is how the
  Brownian-bridge adjustment in `mcop.exotics` is validated.

All formulas assume Black-Scholes dynamics: constant volatility, lognormal
prices, continuous dividend yield q.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

from .black_scholes import bs_price, _d1_d2

__all__ = [
    "digital_cash_or_nothing",
    "digital_asset_or_nothing",
    "geometric_asian_price",
    "geometric_asian_moments",
    "down_and_in_call",
    "down_and_out_call",
]


# --------------------------------------------------------------------------
# Digitals
# --------------------------------------------------------------------------

def digital_cash_or_nothing(
    S: float, K: float, r: float, q: float, sigma: float, T: float,
    is_call: bool, cash: float = 1.0,
) -> float:
    """
    Pays a fixed `cash` amount if the option finishes in the money, else zero.

        call = cash * exp(-rT) * N(d2)
        put  = cash * exp(-rT) * N(-d2)

    N(d2) is the risk-neutral probability of finishing in the money, so a
    cash-or-nothing digital is quite literally a discounted bet on that
    probability. That interpretation is also why delta for a digital behaves so
    badly near the strike at expiry: the payoff is a step function, so its
    derivative tends to a spike.
    """
    if T <= 0.0:
        itm = (S > K) if is_call else (S < K)
        return float(cash) if itm else 0.0
    if sigma <= 0.0:
        fwd = S * math.exp((r - q) * T)
        itm = (fwd > K) if is_call else (fwd < K)
        return float(cash * math.exp(-r * T)) if itm else 0.0

    _, d2, _ = _d1_d2(S, K, r, q, sigma, T)
    prob = ndtr(d2) if is_call else ndtr(-d2)
    return float(cash * math.exp(-r * T) * prob)


def digital_asset_or_nothing(
    S: float, K: float, r: float, q: float, sigma: float, T: float, is_call: bool,
) -> float:
    """
    Pays one unit of the underlying if in the money, else zero.

        call = S * exp(-qT) * N(d1)
        put  = S * exp(-qT) * N(-d1)

    Worth knowing because it decomposes the vanilla exactly:

        vanilla call = asset-or-nothing call - K * cash-or-nothing call

    That identity is asserted in the tests, and it is a neat way to see what the
    two terms of the Black-Scholes formula actually *are*.
    """
    if T <= 0.0:
        itm = (S > K) if is_call else (S < K)
        return float(S) if itm else 0.0
    if sigma <= 0.0:
        fwd = S * math.exp((r - q) * T)
        itm = (fwd > K) if is_call else (fwd < K)
        return float(S * math.exp(-q * T)) if itm else 0.0

    d1, _, _ = _d1_d2(S, K, r, q, sigma, T)
    prob = ndtr(d1) if is_call else ndtr(-d1)
    return float(S * math.exp(-q * T) * prob)


# --------------------------------------------------------------------------
# Geometric Asian
# --------------------------------------------------------------------------

def geometric_asian_moments(
    S: float, r: float, q: float, sigma: float, T: float, n_fixings: int,
) -> tuple[float, float]:
    """
    Mean and variance of log(G), where G is the geometric average of the price
    at n equally spaced fixings t_i = i*T/n for i = 1..n.

    Derivation, because this is the whole reason a closed form exists:

        log S_t = log S0 + (r - q - sigma^2/2) t + sigma W_t

    so log G = (1/n) * sum_i log S_{t_i} is a linear combination of jointly
    normal variables, hence normal. Therefore G is lognormal and the option on
    it prices with a Black-Scholes-shaped formula.

        E[log G]   = log S0 + (r - q - sigma^2/2) * T(n+1)/(2n)
        Var[log G] = sigma^2 * T * (n+1)(2n+1) / (6 n^2)

    The variance term comes from Cov(W_{t_i}, W_{t_j}) = min(t_i, t_j), summed
    over the grid. Note both reduce to the familiar continuous-averaging results
    (T/2 and sigma^2 T/3) as n grows.
    """
    if n_fixings < 1:
        raise ValueError("n_fixings must be at least 1")

    n = float(n_fixings)
    mean_log = math.log(S) + (r - q - 0.5 * sigma**2) * T * (n + 1.0) / (2.0 * n)
    var_log = (sigma**2) * T * (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n * n)
    return mean_log, var_log


def geometric_asian_price(
    S: float, K: float, r: float, q: float, sigma: float, T: float,
    n_fixings: int, is_call: bool,
) -> float:
    """
    Exact price of a fixed-strike geometric-average Asian option under discrete
    monitoring.

    Because log G ~ N(M, V), the expectation of the payoff is the standard
    lognormal result:

        E[(G - K)+] = exp(M + V/2) * N(d1) - K * N(d2)
        d1 = (M - log K + V) / sqrt(V),   d2 = d1 - sqrt(V)

    Discrete monitoring is used rather than the continuous-averaging
    approximation so this matches a simulation on the same fixing grid exactly,
    not just approximately — which is what makes it usable as a control variate.
    """
    if T <= 0.0 or n_fixings < 1:
        raise ValueError("T must be positive and n_fixings at least 1")
    if sigma <= 0.0:
        mean_log, _ = geometric_asian_moments(S, r, q, sigma, T, n_fixings)
        g = math.exp(mean_log)
        intrinsic = max(g - K, 0.0) if is_call else max(K - g, 0.0)
        return float(math.exp(-r * T) * intrinsic)

    M, V = geometric_asian_moments(S, r, q, sigma, T, n_fixings)
    sd = math.sqrt(V)

    d1 = (M - math.log(K) + V) / sd
    d2 = d1 - sd
    fwd = math.exp(M + 0.5 * V)

    if is_call:
        value = fwd * ndtr(d1) - K * ndtr(d2)
    else:
        value = K * ndtr(-d2) - fwd * ndtr(-d1)
    return float(math.exp(-r * T) * value)


# --------------------------------------------------------------------------
# Barriers (continuous monitoring)
# --------------------------------------------------------------------------

def down_and_in_call(
    S: float, K: float, H: float, r: float, q: float, sigma: float, T: float,
) -> float:
    """
    Continuously-monitored down-and-in call, for a barrier at or below the
    strike (H <= K).

        c_di = S e^{-qT} (H/S)^{2L} N(y) - K e^{-rT} (H/S)^{2L-2} N(y - sigma*sqrt(T))

    with L = (r - q + sigma^2/2) / sigma^2 and
    y = log(H^2 / (S K)) / (sigma sqrt(T)) + L sigma sqrt(T).

    The (H/S)^{2L} factor is the reflection principle at work: a barrier problem
    is solved by reflecting the price process in the barrier, and that power term
    is the Radon-Nikodym weight the reflection introduces.

    Restricted to H <= K deliberately. The H > K case takes a different formula,
    and silently applying the wrong one is worse than refusing.
    """
    if H > K:
        raise ValueError(
            f"this formula requires the barrier at or below the strike (H={H} <= K={K}); "
            "the H > K case needs a different expression"
        )
    if H >= S:
        raise ValueError(f"a down-barrier must start below spot (H={H} < S={S})")
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")

    sqrt_T = math.sqrt(T)
    lam = (r - q + 0.5 * sigma**2) / (sigma**2)
    y = math.log(H * H / (S * K)) / (sigma * sqrt_T) + lam * sigma * sqrt_T

    term1 = S * math.exp(-q * T) * (H / S) ** (2.0 * lam) * ndtr(y)
    term2 = K * math.exp(-r * T) * (H / S) ** (2.0 * lam - 2.0) * ndtr(y - sigma * sqrt_T)
    return float(term1 - term2)


def down_and_out_call(
    S: float, K: float, H: float, r: float, q: float, sigma: float, T: float,
) -> float:
    """
    Continuously-monitored down-and-out call, via in-out parity:

        knock-in + knock-out = vanilla

    That identity holds because exactly one of the two pays on every path: either
    the barrier was touched or it was not. It is also the cleanest way to test a
    barrier implementation, since it needs no second closed form — see
    `tests/test_exotics.py`.
    """
    vanilla = bs_price(S, K, r, q, sigma, T, is_call=True)
    return float(vanilla - down_and_in_call(S, K, H, r, q, sigma, T))
