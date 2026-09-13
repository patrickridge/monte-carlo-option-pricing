"""
Market data: price histories, realised volatility, and the implied vol surface.

An honest warning about what is and is not real data here
---------------------------------------------------------
Underlying price history is real and easy to obtain. **Historical option chains
are not** — a full history of quoted bid/ask across every strike and expiry is a
paid dataset (OptionMetrics, CBOE DataShop and similar), and no free source
provides it with enough coverage to backtest against.

This module therefore does something that must be stated plainly rather than
buried, because it is the single biggest caveat on every backtest result the
library produces:

    Option prices in the backtest are **modelled, not observed**. Each option is
    marked with Black–Scholes (or LSM for American exercise) using an implied
    volatility drawn from a parameterised surface, which is itself anchored to
    the realised volatility of the actual underlying price history.

What that buys and what it costs:

- The underlying path is real, so the *directional* behaviour of a strategy —
  when it gets run over, when it profits — is driven by genuine market moves.
- The vol surface is a smooth model. It cannot reproduce a volatility spike that
  front-runs a crash, a bid/ask that gaps to untradeable, or the skew steepening
  that makes put spreads expensive exactly when you want them.
- Consequently, results are best read as **strategy mechanics under a stated vol
  model**, not as a claim about historical P&L. A short-vol strategy will look
  better here than it would in reality, because the model never gaps.

The `variance_risk_premium` parameter is the one deliberate concession to
realism: implied vol trades systematically above subsequent realised vol (the
well-documented VRP), and setting it to zero would make every option
structurally cheap and flatter short-premium strategies even further.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .instruments import OptionContract, DAYS_PER_YEAR

__all__ = [
    "PriceHistory",
    "load_price_csv",
    "synthetic_price_history",
    "fetch_price_history",
    "realized_vol",
    "rolling_realized_vol",
    "ewma_vol",
    "VolSurface",
    "FlatVolSurface",
    "SkewedVolSurface",
    "MarketState",
    "TRADING_DAYS_PER_YEAR",
]

# Volatility is annualised from daily returns by sqrt(252): there are ~252
# trading days in a year, and variance scales linearly in time.
TRADING_DAYS_PER_YEAR = 252.0


@dataclass(frozen=True)
class PriceHistory:
    """
    A daily close series for one underlying.

    Deliberately not a pandas DataFrame: the library's only hard dependencies are
    numpy and scipy, and a date list plus a float array covers everything the
    backtester needs without pulling pandas into the install.
    """

    symbol: str
    dates: list[date]
    closes: np.ndarray

    def __post_init__(self) -> None:
        if len(self.dates) != len(self.closes):
            raise ValueError("dates and closes must have equal length")
        if len(self.dates) < 2:
            raise ValueError("need at least two observations")
        if np.any(self.closes <= 0):
            raise ValueError("closes must be positive")
        if list(self.dates) != sorted(self.dates):
            raise ValueError("dates must be sorted ascending")

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def log_returns(self) -> np.ndarray:
        """Daily log returns, length len(self) - 1."""
        return np.diff(np.log(self.closes))

    def index_of(self, d: date) -> int:
        """Index of the first observation on or after `d`."""
        for i, obs in enumerate(self.dates):
            if obs >= d:
                return i
        raise KeyError(f"no observation on or after {d}")

    def close_on(self, d: date) -> float:
        return float(self.closes[self.index_of(d)])

    def slice(self, start: date | None = None, end: date | None = None) -> "PriceHistory":
        lo = self.index_of(start) if start else 0
        hi = (self.index_of(end) + 1) if end else len(self)
        return PriceHistory(self.symbol, self.dates[lo:hi], self.closes[lo:hi])


def load_price_csv(path: str | Path, symbol: str | None = None) -> PriceHistory:
    """
    Load a daily close series from CSV.

    Accepts the common shapes emitted by Yahoo Finance, Stooq and broker
    exports: a date column named any of date/Date/timestamp, and a close column
    named any of close/Close/adj_close/Adj Close. Adjusted close is preferred
    when present, because splits and dividends otherwise show up as fake
    overnight gaps that a vol estimator would read as real risk.
    """
    path = Path(path)
    date_keys = ("date", "Date", "timestamp", "Timestamp", "time")
    close_keys = ("adj_close", "Adj Close", "adjclose", "AdjClose", "close", "Close")

    dates: list[date] = []
    closes: list[float] = []

    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")

        date_col = next((k for k in date_keys if k in reader.fieldnames), None)
        close_col = next((k for k in close_keys if k in reader.fieldnames), None)
        if date_col is None or close_col is None:
            raise ValueError(
                f"{path}: could not find a date column (tried {date_keys}) and a close "
                f"column (tried {close_keys}). Found: {reader.fieldnames}"
            )

        for row in reader:
            raw_date, raw_close = row[date_col], row[close_col]
            if not raw_date or not raw_close:
                continue  # skip blank rows rather than crashing the load
            dates.append(date.fromisoformat(raw_date[:10]))
            closes.append(float(raw_close))

    order = np.argsort(np.array([d.toordinal() for d in dates]))
    dates = [dates[i] for i in order]
    closes_arr = np.array(closes, dtype=float)[order]

    return PriceHistory(symbol or path.stem, dates, closes_arr)


def synthetic_price_history(
    symbol: str = "SYNTH",
    S0: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.20,
    n_days: int = 756,
    start: date = date(2022, 1, 3),
    seed: int | None = 42,
    vol_of_vol: float = 0.0,
) -> PriceHistory:
    """
    Generate a reproducible synthetic price history under GBM.

    This exists so the entire strategy and backtest stack is runnable with no
    network access and no paid data, which matters for tests and for anyone
    cloning the repo. Note the drift is under the **real-world** measure here,
    not the risk-neutral one: a backtest is asking what a strategy would have
    earned, so the underlying should drift at its actual expected return, not at
    r. Pricing inside the backtest still discounts at r — mixing those two
    measures up is a classic error and the separation is deliberate.

    Set `vol_of_vol` above zero to let volatility itself wander (a crude
    stochastic-vol proxy), which stops short-premium strategies from looking
    artificially safe under perfectly constant vol.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS_PER_YEAR

    if vol_of_vol > 0:
        # Log-AR(1) vol process, mean-reverting to sigma.
        log_vol = np.empty(n_days)
        log_vol[0] = math.log(sigma)
        phi = 0.98
        for i in range(1, n_days):
            log_vol[i] = (1 - phi) * math.log(sigma) + phi * log_vol[i - 1] + vol_of_vol * math.sqrt(dt) * rng.standard_normal()
        vols = np.exp(log_vol)
    else:
        vols = np.full(n_days, sigma)

    shocks = rng.standard_normal(n_days)
    log_rets = (mu - 0.5 * vols**2) * dt + vols * math.sqrt(dt) * shocks

    closes = np.empty(n_days + 1)
    closes[0] = S0
    closes[1:] = S0 * np.exp(np.cumsum(log_rets))

    # Business days only, so weekends do not appear as tradeable dates.
    dates: list[date] = []
    d = start
    while len(dates) < n_days + 1:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)

    return PriceHistory(symbol, dates, closes)


def fetch_price_history(symbol: str, start: date, end: date) -> PriceHistory:
    """
    Download real daily closes via yfinance, if it is installed.

    Kept as an optional extra rather than a dependency: the library must work
    offline, and a hard dependency on a scraping package that breaks whenever
    Yahoo changes its API is not something a pricing library should carry.
    """
    try:
        import yfinance  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise ImportError(
            "fetch_price_history needs yfinance: pip install yfinance. "
            "Alternatively use load_price_csv() with an exported file, or "
            "synthetic_price_history() to run offline."
        ) from exc

    data = yfinance.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if data is None or len(data) == 0:  # pragma: no cover - network dependent
        raise ValueError(f"no data returned for {symbol} between {start} and {end}")

    closes = np.asarray(data["Close"].values, dtype=float).reshape(-1)
    dates = [d.date() for d in data.index.to_pydatetime()]
    return PriceHistory(symbol, dates, closes)


def realized_vol(log_returns: np.ndarray, annualise: bool = True) -> float:
    """
    Close-to-close realised volatility.

    Uses the zero-mean estimator (sqrt of mean squared return) rather than
    subtracting the sample mean. Over the short windows used for vol estimation
    the sample mean is almost pure noise, and subtracting it adds variance to
    the estimate without removing meaningful bias.
    """
    r = np.asarray(log_returns, dtype=float)
    if r.size == 0:
        return float("nan")
    vol = float(np.sqrt(np.mean(np.square(r))))
    return vol * math.sqrt(TRADING_DAYS_PER_YEAR) if annualise else vol


def rolling_realized_vol(history: PriceHistory, window: int = 21) -> np.ndarray:
    """
    Rolling realised vol aligned to `history.dates`.

    The value at index i uses only returns strictly before i, so the series is
    usable inside a backtest without look-ahead. Entries with insufficient
    history are NaN and callers must handle them — silently back-filling would
    be exactly the kind of lookahead this alignment exists to prevent.
    """
    rets = history.log_returns  # rets[i] is the return from date i to date i+1
    out = np.full(len(history), np.nan)
    for i in range(len(history)):
        # Returns realised strictly before date i are rets[:i].
        if i < window:
            continue
        out[i] = realized_vol(rets[i - window:i])
    return out


def ewma_vol(log_returns: np.ndarray, lam: float = 0.94) -> float:
    """
    Exponentially weighted realised vol (RiskMetrics convention, lambda = 0.94).

    Reacts faster than an equal-weighted window: after a shock, an EWMA vol
    rises immediately, whereas a 21-day window steps up and then drops
    mechanically 21 days later when the shock leaves the window regardless of
    what the market is doing.
    """
    r = np.asarray(log_returns, dtype=float)
    if r.size == 0:
        return float("nan")
    weights = lam ** np.arange(r.size - 1, -1, -1)
    weights /= weights.sum()
    var = float(np.sum(weights * np.square(r)))
    return math.sqrt(var * TRADING_DAYS_PER_YEAR)


class VolSurface:
    """Interface: map (strike, spot, time to expiry) to an implied volatility."""

    def iv(self, strike: float, spot: float, T: float, is_call: bool = True) -> float:
        raise NotImplementedError

    def for_contract(self, contract: OptionContract, spot: float, asof: date) -> float:
        return self.iv(contract.strike, spot, contract.time_to_expiry(asof), contract.is_call)


@dataclass
class FlatVolSurface(VolSurface):
    """Constant vol across all strikes and maturities. Useful as a control."""

    level: float = 0.20

    def iv(self, strike: float, spot: float, T: float, is_call: bool = True) -> float:
        return float(self.level)


@dataclass
class SkewedVolSurface(VolSurface):
    """
    A parameterised equity-style implied vol surface.

        iv(K, T) = atm(T) + skew * k + curvature * k^2,   k = ln(K / S)

    with atm(T) = atm_level + term_slope * (sqrt(T) - sqrt(T_ref)).

    Shape rationale — this is not arbitrary, it reproduces the two robust
    features of a real equity surface:

    - **Negative skew** (`skew < 0`): low-strike puts trade at higher implied
      vol than high-strike calls. Crash risk is one-sided in equities, and
      demand for downside protection is structural. Getting this sign right is
      what makes put spreads correctly expensive relative to call spreads.
    - **Upward term structure** (`term_slope > 0`): longer-dated options carry
      more vol premium in calm regimes. Set negative to model an inverted,
      stressed surface.

    What it deliberately does not model: vol spikes correlated with price
    drops, and any dynamic evolution of skew. Both would flatter or punish
    specific strategies in ways this smooth model cannot capture, which is why
    the module docstring insists results are mechanics under a stated model
    rather than historical truth.
    """

    atm_level: float = 0.20
    skew: float = -0.30
    curvature: float = 0.50
    term_slope: float = 0.02
    T_ref: float = 30.0 / DAYS_PER_YEAR
    floor: float = 0.01
    cap: float = 3.00

    def iv(self, strike: float, spot: float, T: float, is_call: bool = True) -> float:
        if spot <= 0 or strike <= 0:
            raise ValueError("spot and strike must be positive")
        T_eff = max(T, 1.0 / DAYS_PER_YEAR)  # a same-day option still needs a vol

        k = math.log(strike / spot)
        atm = self.atm_level + self.term_slope * (math.sqrt(T_eff) - math.sqrt(self.T_ref))
        vol = atm + self.skew * k + self.curvature * k * k
        return float(min(max(vol, self.floor), self.cap))

    def with_atm(self, atm_level: float) -> "SkewedVolSurface":
        """
        Return a copy re-anchored to a new ATM level.

        The backtester calls this daily to re-anchor the surface to trailing
        realised vol plus the variance risk premium, so the shape stays fixed
        while the level tracks the market.
        """
        return SkewedVolSurface(
            atm_level=atm_level,
            skew=self.skew,
            curvature=self.curvature,
            term_slope=self.term_slope,
            T_ref=self.T_ref,
            floor=self.floor,
            cap=self.cap,
        )


@dataclass
class MarketState:
    """
    Everything needed to value a position on one date.

    Bundling these together rather than passing five loose floats means a
    valuation cannot accidentally mix today's spot with yesterday's rate, and it
    gives the backtester one object to advance per day.
    """

    asof: date
    spot: float
    r: float = 0.04
    q: float = 0.0
    vol_surface: VolSurface = field(default_factory=SkewedVolSurface)

    def iv_for(self, contract: OptionContract) -> float:
        return self.vol_surface.for_contract(contract, self.spot, self.asof)

    def with_spot(self, spot: float) -> "MarketState":
        """Copy at a different spot. Used to sweep value across a price grid."""
        return MarketState(self.asof, spot, self.r, self.q, self.vol_surface)

    def with_date(self, asof: date) -> "MarketState":
        return MarketState(asof, self.spot, self.r, self.q, self.vol_surface)
