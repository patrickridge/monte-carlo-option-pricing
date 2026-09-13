"""
Analytic Black–Scholes–Merton pricing, Greeks, and implied volatility.

Why this module exists
----------------------
The Monte Carlo engine is the point of this project, but a closed-form pricer
earns its place for three reasons:

1. **Validation.** MC Greeks are noisy estimators. Analytic European Greeks give
   an exact reference to check them against (see `tests/test_greeks_vs_bs.py`).
2. **Speed.** The backtester reprices every leg of every open position on every
   trading day. Running LSM for each of those marks would take hours; the
   closed form takes microseconds. European legs are marked analytically and
   only American legs fall back to Monte Carlo.
3. **Implied volatility.** Inverting the pricer for sigma is how a quoted option
   price becomes a vol number, which is what the strategy layer actually reasons
   about.

Sign and scaling conventions (these matter, and shops differ)
-------------------------------------------------------------
All functions here return **raw partial derivatives in natural units**:

- ``delta``  = dV/dS            per 1.00 of spot
- ``gamma``  = d2V/dS2          per 1.00 of spot, squared
- ``vega``   = dV/dsigma        per 1.00 of vol (i.e. per 100 vol points)
- ``theta``  = dV/dt            per **year**, and negative for a long option
                                (value decays as calendar time advances)
- ``rho``    = dV/dr            per 1.00 of rate

Trading desks usually quote vega per 1 vol point (vega / 100) and theta per
calendar day (theta / 365). `Greeks.scaled()` does that conversion explicitly
rather than baking it in, so the raw numbers stay comparable to a finite
difference of the pricer.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.special import ndtr  # vectorised standard normal CDF

__all__ = [
    "Greeks",
    "bs_price",
    "bs_greeks",
    "implied_vol",
    "forward_price",
]

# Standard normal PDF. scipy has this via norm.pdf but the closed form avoids
# the (surprisingly heavy) scipy.stats import at module load.
_INV_SQRT_2PI = 0.3989422804014327


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * np.square(x))


def forward_price(S: float, r: float, q: float, T: float) -> float:
    """Forward price of the underlying: F = S * exp((r - q) T)."""
    return S * np.exp((r - q) * T)


def _d1_d2(S, K, r, q, sigma, T):
    """
    Standard BSM d1/d2 terms.

    Returns (d1, d2, sqrt_T). Caller is responsible for handling the degenerate
    cases (T <= 0 or sigma <= 0) where these are not finite.
    """
    sqrt_T = np.sqrt(T)
    vol_sqrt_T = sigma * sqrt_T
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    return d1, d2, sqrt_T


@dataclass(frozen=True)
class Greeks:
    """
    A price and its risk sensitivities, in the raw units documented above.

    `price` is carried alongside the sensitivities because every method that
    produces Greeks computes the price on the way, and returning it separately
    invites the two to be computed with inconsistent inputs.
    """

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def scaled(self, vol_points: bool = True, theta_per_day: bool = True) -> "Greeks":
        """
        Convert to desk-quoting conventions.

        vega_points   : vega per 1 vol point (a move from 20% to 21%)
        theta_per_day : theta per calendar day rather than per year
        """
        return Greeks(
            price=self.price,
            delta=self.delta,
            gamma=self.gamma,
            vega=self.vega / 100.0 if vol_points else self.vega,
            theta=self.theta / 365.0 if theta_per_day else self.theta,
            rho=self.rho / 100.0 if vol_points else self.rho,
        )

    def __mul__(self, k: float) -> "Greeks":
        """Scale by a position size. Used to aggregate legs into a portfolio."""
        return Greeks(
            price=self.price * k,
            delta=self.delta * k,
            gamma=self.gamma * k,
            vega=self.vega * k,
            theta=self.theta * k,
            rho=self.rho * k,
        )

    __rmul__ = __mul__

    def __add__(self, other: "Greeks") -> "Greeks":
        """Greeks are additive across legs. This is what makes netting work."""
        return Greeks(
            price=self.price + other.price,
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            vega=self.vega + other.vega,
            theta=self.theta + other.theta,
            rho=self.rho + other.rho,
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def bs_price(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    is_call: bool,
) -> float:
    """
    Black–Scholes–Merton price of a European option with continuous dividend
    yield q.

    Degenerate inputs are handled rather than rejected, because the backtester
    walks positions all the way into expiry and will legitimately ask for a
    price at T = 0:

    - T <= 0      -> intrinsic value at spot
    - sigma <= 0  -> deterministic forward, discounted intrinsic
    """
    S = float(S)
    K = float(K)

    if T <= 0.0:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return float(intrinsic)

    if sigma <= 0.0:
        fwd = forward_price(S, r, q, T)
        intrinsic = max(fwd - K, 0.0) if is_call else max(K - fwd, 0.0)
        return float(np.exp(-r * T) * intrinsic)

    d1, d2, _ = _d1_d2(S, K, r, q, sigma, T)
    df_r = np.exp(-r * T)
    df_q = np.exp(-q * T)

    if is_call:
        price = S * df_q * ndtr(d1) - K * df_r * ndtr(d2)
    else:
        price = K * df_r * ndtr(-d2) - S * df_q * ndtr(-d1)
    return float(price)


def bs_greeks(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    is_call: bool,
) -> Greeks:
    """
    Closed-form BSM Greeks.

    Theta is returned per year and carries the usual sign: a long vanilla option
    with no rate effects has theta < 0, because the derivative is taken with
    respect to *calendar time advancing*, not with respect to T. The two differ
    by a minus sign and conflating them is a classic source of sign bugs, so the
    formula below computes dV/dt directly.

    At T <= 0 the option is dead: price is intrinsic and every sensitivity is
    zero (gamma and vega are genuinely singular in the limit, but a dead
    contract carries no risk, which is what a risk report needs to show).
    """
    if T <= 0.0:
        return Greeks(
            price=bs_price(S, K, r, q, sigma, T, is_call),
            delta=0.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            rho=0.0,
        )

    if sigma <= 0.0:
        # Deterministic: delta is the discounted forward sensitivity, and there
        # is no convexity or vol exposure to report.
        fwd = forward_price(S, r, q, T)
        itm = (fwd > K) if is_call else (fwd < K)
        sign = 1.0 if is_call else -1.0
        return Greeks(
            price=bs_price(S, K, r, q, sigma, T, is_call),
            delta=float(sign * np.exp(-q * T)) if itm else 0.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            rho=0.0,
        )

    d1, d2, sqrt_T = _d1_d2(S, K, r, q, sigma, T)
    df_r = np.exp(-r * T)
    df_q = np.exp(-q * T)
    pdf_d1 = _norm_pdf(d1)

    # Gamma and vega are identical for calls and puts (put-call parity is linear
    # in S, so it drops out of the second derivative and the vol derivative).
    gamma = df_q * pdf_d1 / (S * sigma * sqrt_T)
    vega = S * df_q * pdf_d1 * sqrt_T

    # Theta = dV/dt. Three contributions: the gamma/decay term, the carry on the
    # stock leg (q), and the financing on the strike leg (r).
    decay_term = -(S * df_q * pdf_d1 * sigma) / (2.0 * sqrt_T)

    if is_call:
        delta = df_q * ndtr(d1)
        theta = decay_term + q * S * df_q * ndtr(d1) - r * K * df_r * ndtr(d2)
        rho = K * T * df_r * ndtr(d2)
    else:
        delta = -df_q * ndtr(-d1)
        theta = decay_term - q * S * df_q * ndtr(-d1) + r * K * df_r * ndtr(-d2)
        rho = -K * T * df_r * ndtr(-d2)

    return Greeks(
        price=bs_price(S, K, r, q, sigma, T, is_call),
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )


def implied_vol(
    price: float,
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    is_call: bool,
    tol: float = 1e-8,
    max_iter: int = 100,
    lo: float = 1e-6,
    hi: float = 5.0,
    xtol: float = 1e-10,
) -> float:
    """
    Recover the volatility that reproduces an observed option price.

    Uses bisection on [lo, hi] rather than Newton. Newton converges faster but
    divides by vega, which collapses toward zero for deep in- or out-of-the-money
    options — exactly the quotes where an IV solver gets used in anger and
    exactly where Newton diverges. Bisection is slower and unconditionally
    robust; ~40 iterations is not the bottleneck anywhere this is called.

    Convergence is measured on the **width of the volatility bracket** (`xtol`),
    not on the price residual. That distinction matters: for a deep in- or
    out-of-the-money option vega is tiny, so a price residual of 1e-8 can still
    leave the volatility wrong in the fourth decimal. Terminating on price would
    silently return a badly imprecise vol exactly where precision is hardest,
    which is the wrong failure mode for a quantity other code then trades on.

    Returns NaN when the price is outside the no-arbitrage band, because there is
    no volatility that explains it. That is a data-quality signal the caller
    should see, not an exception to swallow: a stale or crossed quote should skip
    a trade, not halt a backtest.
    """
    if T <= 0.0:
        return float("nan")

    # No-arbitrage bounds. Below the lower bound or above the upper bound, no
    # sigma reproduces the price.
    df_r = np.exp(-r * T)
    df_q = np.exp(-q * T)
    if is_call:
        lower = max(S * df_q - K * df_r, 0.0)
        upper = S * df_q
    else:
        lower = max(K * df_r - S * df_q, 0.0)
        upper = K * df_r

    if not (lower - tol <= price <= upper + tol):
        return float("nan")

    f_lo = bs_price(S, K, r, q, lo, T, is_call) - price
    f_hi = bs_price(S, K, r, q, hi, T, is_call) - price

    # Price is monotonically increasing in sigma, so a sign change must bracket
    # the root. No sign change means the target sits outside [lo, hi].
    if f_lo > 0.0:
        return float(lo)
    if f_hi < 0.0:
        return float("nan")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if (hi - lo) < xtol:
            return float(mid)
        # Price is increasing in sigma, so the sign of the residual says which
        # half of the bracket still contains the root.
        if bs_price(S, K, r, q, mid, T, is_call) - price < 0.0:
            lo = mid
        else:
            hi = mid

    return float(0.5 * (lo + hi))
