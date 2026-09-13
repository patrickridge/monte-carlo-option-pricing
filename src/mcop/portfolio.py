"""
Valuation and risk aggregation for multi-leg positions.

This is where the pricers become usable for position management. The central
fact it relies on is that **Greeks are additive**: the delta of a spread is the
sum of the deltas of its legs, each scaled by signed size. That is why
`instruments.Leg` carries a signed quantity and a multiplier — netting is then a
plain sum with no branching on long/short, which is where sign errors normally
creep in.

Which pricer marks which leg
-----------------------------
- **European** legs: closed-form Black–Scholes.
- **American** legs: the CRR binomial tree, *not* Longstaff–Schwartz.

That second choice is deliberate and worth defending, because LSM is the
headline feature of this library. A backtest reprices every leg of every open
position on every trading day — tens of thousands of valuations. Two problems
with using Monte Carlo for that:

1. **Speed.** LSM at 50k paths takes ~0.5s per valuation. A 3-year daily
   backtest would take hours. The binomial tree takes ~1ms.
2. **Noise, which is the real problem.** MC marks carry sampling error. Marking
   the same unchanged position on two consecutive days would produce different
   values purely from the random draw, injecting fake P&L volatility into the
   equity curve and corrupting every risk statistic computed from it — Sharpe
   most of all. A deterministic pricer means a P&L change reflects a market
   change and nothing else.

Monte Carlo remains the right tool for validation and for payoffs the tree
cannot handle; `mc_price_leg` exposes it. The tree and LSM agree to well inside
MC error (`tests/test_american_put_vs_binomial.py`), so this is a
speed/determinism swap, not an accuracy compromise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import Greeks, bs_price, bs_greeks
from .binomial_tree import american_option_crr
from .instruments import DAYS_PER_YEAR, Leg, OptionContract, Position, Underlying
from .market_data import MarketState
from .simulate_paths import simulate_gbm_paths
from .american_lsm import american_option_lsm

__all__ = [
    "CRR_STEPS",
    "price_instrument",
    "leg_value",
    "leg_greeks",
    "position_value",
    "position_greeks",
    "value_curve",
    "profit_curve",
    "breakevens",
    "mc_price_leg",
    "PositionReport",
    "report",
]

# Binomial steps for American marks. 200 puts the tree's discretisation error
# well below a penny for typical equity options while staying ~1ms per call.
CRR_STEPS = 200


def price_instrument(
    instrument: OptionContract | Underlying,
    market: MarketState,
    crr_steps: int = CRR_STEPS,
    vol_override: float | None = None,
) -> float:
    """
    Price one unit of an instrument (per share, before any multiplier).

    `vol_override` bypasses the surface, which the Greek bumps below need in
    order to shock vol while holding everything else fixed.
    """
    if isinstance(instrument, Underlying):
        return float(market.spot)

    T = instrument.time_to_expiry(market.asof)
    sigma = market.iv_for(instrument) if vol_override is None else vol_override

    # An expired contract is worth its intrinsic value; both pricers below
    # require T > 0, so this case is handled before dispatching.
    if T <= 0.0:
        return float(instrument.intrinsic(market.spot))

    if instrument.style.value == "european":
        return bs_price(market.spot, instrument.strike, market.r, market.q, sigma, T, instrument.is_call)

    return float(
        american_option_crr(
            S0=market.spot,
            K=instrument.strike,
            r=market.r,
            sigma=sigma,
            T=T,
            n_steps=crr_steps,
            is_call=instrument.is_call,
            q=market.q,
        )
    )


def _american_greeks(
    contract: OptionContract,
    market: MarketState,
    crr_steps: int,
    rel_bump: float = 0.01,
    vol_bump: float = 0.01,
    t_bump_days: float = 1.0,
    r_bump: float = 1e-4,
) -> Greeks:
    """
    Finite-difference Greeks for an American option off the binomial tree.

    The tree is deterministic, so no common-random-numbers trick is needed — the
    only error is discretisation, and central differences cancel the leading
    term. Theta uses a one-day bump because that is both the natural quoting
    unit and large enough that tree granularity does not dominate the
    difference.
    """
    S = market.spot
    T = contract.time_to_expiry(market.asof)
    sigma = market.iv_for(contract)

    if T <= 0.0:
        return Greeks(contract.intrinsic(S), 0.0, 0.0, 0.0, 0.0, 0.0)

    def px(S_=S, sigma_=sigma, T_=T, r_=market.r) -> float:
        if T_ <= 0:
            return float(contract.intrinsic(S_))
        return float(
            american_option_crr(S_, contract.strike, r_, sigma_, T_, crr_steps, contract.is_call, market.q)
        )

    base = px()
    h_S = rel_bump * S
    up, down = px(S_=S + h_S), px(S_=S - h_S)

    delta = (up - down) / (2.0 * h_S)
    gamma = (up - 2.0 * base + down) / (h_S**2)
    vega = (px(sigma_=sigma + vol_bump) - px(sigma_=sigma - vol_bump)) / (2.0 * vol_bump)

    # Theta as the value change per year from calendar time advancing.
    h_T = min(t_bump_days / DAYS_PER_YEAR, 0.5 * T)
    theta = -(px(T_=T + h_T) - px(T_=T - h_T)) / (2.0 * h_T)

    rho = (px(r_=market.r + r_bump) - px(r_=market.r - r_bump)) / (2.0 * r_bump)

    return Greeks(price=base, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def instrument_greeks(
    instrument: OptionContract | Underlying,
    market: MarketState,
    crr_steps: int = CRR_STEPS,
) -> Greeks:
    """Greeks for one unit of an instrument, before position scaling."""
    if isinstance(instrument, Underlying):
        # Stock is linear in itself: delta 1, no convexity, no vol or time decay.
        return Greeks(price=market.spot, delta=1.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    T = instrument.time_to_expiry(market.asof)
    if T <= 0.0:
        return Greeks(instrument.intrinsic(market.spot), 0.0, 0.0, 0.0, 0.0, 0.0)

    if instrument.style.value == "european":
        return bs_greeks(
            market.spot, instrument.strike, market.r, market.q,
            market.iv_for(instrument), T, instrument.is_call,
        )
    return _american_greeks(instrument, market, crr_steps)


def leg_value(leg: Leg, market: MarketState, crr_steps: int = CRR_STEPS) -> float:
    """Signed currency value of a leg, including multiplier and direction."""
    return leg.signed_shares * price_instrument(leg.instrument, market, crr_steps)


def leg_greeks(leg: Leg, market: MarketState, crr_steps: int = CRR_STEPS) -> Greeks:
    """
    Position-scaled Greeks for a leg.

    Scaling by `signed_shares` is what makes a short leg carry negative gamma
    and positive theta automatically, with no special-casing anywhere.
    """
    return instrument_greeks(leg.instrument, market, crr_steps) * leg.signed_shares


def position_value(position: Position, market: MarketState, crr_steps: int = CRR_STEPS) -> float:
    """Mark-to-market value of the whole position."""
    return float(sum(leg_value(leg, market, crr_steps) for leg in position.legs))


def position_greeks(position: Position, market: MarketState, crr_steps: int = CRR_STEPS) -> Greeks:
    """Netted Greeks across every leg — the risk report for the position."""
    total = Greeks(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for leg in position.legs:
        total = total + leg_greeks(leg, market, crr_steps)
    return total


def value_curve(
    position: Position,
    market: MarketState,
    spots: np.ndarray,
    crr_steps: int = CRR_STEPS,
) -> np.ndarray:
    """
    Mark the position across a grid of underlying prices, holding the valuation
    date fixed.

    This is the general form of a payoff diagram and the one to use when legs
    have different expiries or when you want the curve *before* expiry, where
    time value rounds off the kinks. Every point is a full reprice, so it is
    slower than `Position.payoff_at_expiry` — which is exact but only valid at a
    shared expiry.
    """
    spots = np.asarray(spots, dtype=float)
    return np.array([position_value(position, market.with_spot(float(s)), crr_steps) for s in spots])


def profit_curve(
    position: Position,
    spots: np.ndarray,
    entry_cost: float,
    market: MarketState | None = None,
    crr_steps: int = CRR_STEPS,
) -> np.ndarray:
    """
    Profit and loss across underlying prices.

    `entry_cost` is the signed cash paid to open: positive for a net debit
    (you paid), negative for a net credit (you received). Profit is therefore
    value - entry_cost, which is correct for both.

    With `market=None` the curve is at expiry (exact, requires a single expiry);
    pass a `MarketState` to get the curve on a live date instead.
    """
    if market is None:
        value = position.payoff_at_expiry(spots)
    else:
        value = value_curve(position, market, spots, crr_steps)
    return np.asarray(value, dtype=float) - entry_cost


def breakevens(
    position: Position,
    entry_cost: float,
    spot_lo: float | None = None,
    spot_hi: float | None = None,
    n_grid: int = 4001,
) -> list[float]:
    """
    Underlying prices at which the position breaks even at expiry.

    Found by scanning a dense grid for sign changes in P&L and refining each
    with bisection. A grid scan rather than an analytic solve because a general
    multi-leg payoff is piecewise linear with an arbitrary number of kinks, so
    the number of roots is not known in advance — an iron condor has two, a
    butterfly two, a naked call one, a calendar none at a shared expiry.
    """
    strikes = [leg.instrument.strike for leg in position.option_legs]
    if not strikes:
        raise ValueError("breakevens needs at least one option leg")

    lo = spot_lo if spot_lo is not None else max(0.01, 0.2 * min(strikes))
    hi = spot_hi if spot_hi is not None else 2.0 * max(strikes)

    grid = np.linspace(lo, hi, n_grid)
    pnl = np.asarray(position.payoff_at_expiry(grid), dtype=float) - entry_cost

    roots: list[float] = []
    for i in range(len(grid) - 1):
        a, b = pnl[i], pnl[i + 1]
        if a == 0.0:
            roots.append(float(grid[i]))
        elif a * b < 0.0:
            # Linear interpolation is exact here: the payoff is piecewise linear
            # and a sign change on a fine grid lies within a single segment.
            x = grid[i] + (grid[i + 1] - grid[i]) * (0.0 - a) / (b - a)
            roots.append(float(x))

    # Deduplicate near-identical roots from adjacent cells.
    out: list[float] = []
    for x in sorted(roots):
        if not out or abs(x - out[-1]) > 1e-6:
            out.append(x)
    return out


def mc_price_leg(
    leg: Leg,
    market: MarketState,
    n_paths: int = 50_000,
    n_steps: int = 100,
    seed: int | None = 123,
    degree: int = 2,
) -> float:
    """
    Mark a leg by Monte Carlo instead of the analytic/tree pricer.

    Not used in the backtest loop (see the module docstring on noise), but it is
    the cross-check that the fast marks are right, and the path a genuinely
    path-dependent payoff would take.
    """
    if isinstance(leg.instrument, Underlying):
        return leg.signed_shares * market.spot

    contract = leg.instrument
    T = contract.time_to_expiry(market.asof)
    if T <= 0.0:
        return leg.signed_shares * contract.intrinsic(market.spot)

    sigma = market.iv_for(contract)
    paths = simulate_gbm_paths(
        S0=market.spot, r=market.r, sigma=sigma, T=T,
        n_steps=n_steps, n_paths=n_paths, q=market.q, seed=seed, antithetic=True,
    )

    if contract.style.value == "american":
        px = american_option_lsm(paths, K=contract.strike, r=market.r, T=T,
                                 is_call=contract.is_call, degree=degree, q=market.q)
    else:
        ST = paths[:, -1]
        payoff = np.maximum(ST - contract.strike, 0.0) if contract.is_call else np.maximum(contract.strike - ST, 0.0)
        px = float(np.exp(-market.r * T) * payoff.mean())

    return leg.signed_shares * float(px)


@dataclass
class PositionReport:
    """A position's value and netted risk on one date, ready to print."""

    name: str
    asof: str
    spot: float
    value: float
    greeks: Greeks

    def __str__(self) -> str:
        g = self.greeks.scaled()  # vega per vol point, theta per day
        return (
            f"{self.name} @ {self.asof}  spot={self.spot:.2f}\n"
            f"  value : {self.value:>12,.2f}\n"
            f"  delta : {self.greeks.delta:>12,.2f}  (share-equivalent exposure)\n"
            f"  gamma : {self.greeks.gamma:>12,.4f}  (delta change per 1.00 move)\n"
            f"  vega  : {g.vega:>12,.2f}  (per 1 vol point)\n"
            f"  theta : {g.theta:>12,.2f}  (per calendar day)\n"
            f"  rho   : {g.rho:>12,.2f}  (per 1% rate move)"
        )


def report(position: Position, market: MarketState, crr_steps: int = CRR_STEPS) -> PositionReport:
    """Bundle value and netted Greeks for display."""
    return PositionReport(
        name=position.name,
        asof=market.asof.isoformat(),
        spot=market.spot,
        value=position_value(position, market, crr_steps),
        greeks=position_greeks(position, market, crr_steps),
    )
