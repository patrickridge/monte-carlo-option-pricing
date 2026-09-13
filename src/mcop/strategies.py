"""
Strategy templates: named multi-leg structures built from `instruments.Leg`.

Every builder here is a thin function returning a `Position`. That thinness is
the point — once legs net correctly (`mcop.portfolio`), a covered call is
genuinely just "long 100 shares, short 1 call", and an iron condor is four legs
with the right signs. Encoding them as one-line compositions rather than
bespoke classes means each structure inherits valuation, Greeks, payoff
diagrams and P&L for free, and there is no per-strategy pricing code to get
subtly wrong.

Specifying strikes: by delta, not by price
-------------------------------------------
Traders do not say "sell the 112.50 call"; they say "sell the 25-delta call".
Delta is roughly the risk-neutral probability of finishing in the money, so a
delta target expresses a consistent amount of risk across different spots,
vols, and maturities, whereas a fixed strike or a fixed percentage offset does
not: 5% out of the money is a near-certain expiry in a quiet market and a
coin-flip in a volatile one.

`strike_for_delta` inverts the pricer to find the strike matching a target
delta, which is what makes the backtester's entries comparable across a
multi-year run with changing volatility.
"""

from __future__ import annotations

from datetime import date

from .black_scholes import bs_greeks
from .instruments import Leg, OptionContract, Position, Right, Style, Underlying
from .market_data import MarketState

__all__ = [
    "option",
    "strike_for_delta",
    "round_to_increment",
    "covered_call",
    "protective_put",
    "collar",
    "vertical_spread",
    "straddle",
    "strangle",
    "iron_condor",
    "butterfly",
    "calendar_spread",
    "short_strangle_by_delta",
    "iron_condor_by_delta",
    "covered_call_by_delta",
]


def option(
    underlying: str,
    right: Right | str,
    strike: float,
    expiry: date,
    style: Style | str = Style.AMERICAN,
    multiplier: float = 100.0,
) -> OptionContract:
    """Terse contract constructor, so builders below read as trade descriptions."""
    return OptionContract(
        underlying=underlying,
        right=Right(right),
        strike=float(strike),
        expiry=expiry,
        style=Style(style),
        multiplier=multiplier,
    )


def round_to_increment(strike: float, increment: float = 1.0) -> float:
    """
    Snap to a listed strike.

    Real chains are not continuous — equities list $1, $2.50 or $5 increments
    depending on price. Backtesting against continuous strikes quietly assumes
    a perfect fit at any delta and overstates how precisely a target can be hit.
    """
    if increment <= 0:
        return float(strike)
    return float(round(strike / increment) * increment)


def strike_for_delta(
    market: MarketState,
    target_delta: float,
    expiry: date,
    is_call: bool,
    increment: float = 1.0,
    lo_mult: float = 0.2,
    hi_mult: float = 3.0,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """
    Find the strike whose delta matches `target_delta` (given as a magnitude,
    e.g. 0.25 for a 25-delta option).

    Solved by bisection on strike. Delta is monotone in strike — a call's delta
    falls from ~1 deep ITM toward 0 far OTM — so a sign change brackets the
    root. Bisection also copes with the volatility surface making delta a
    slightly awkward function of strike: because skew means vol *itself* varies
    with strike, delta is not the clean closed form it would be under flat vol,
    and a derivative-based solve would need the surface's slope. Bisection needs
    only monotonicity.

    Note the vol used at each candidate strike is read from the surface, so the
    resulting strike is skew-consistent rather than computed off a single ATM
    vol.
    """
    target = abs(target_delta)
    if not (0.0 < target < 1.0):
        raise ValueError("target_delta must be strictly between 0 and 1")

    T = max((expiry - market.asof).days, 0) / 365.0
    if T <= 0:
        raise ValueError("expiry must be in the future")

    def abs_delta(K: float) -> float:
        contract = option(symbol_of(market), Right.CALL if is_call else Right.PUT, K, expiry, Style.EUROPEAN)
        sigma = market.iv_for(contract)
        g = bs_greeks(market.spot, K, market.r, market.q, sigma, T, is_call)
        return abs(g.delta)

    lo, hi = lo_mult * market.spot, hi_mult * market.spot

    # abs(delta) is decreasing in strike for calls and increasing for puts.
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        d = abs_delta(mid)
        if abs(d - target) < tol or (hi - lo) < tol:
            break
        if is_call:
            # Higher strike -> smaller call delta.
            if d > target:
                lo = mid
            else:
                hi = mid
        else:
            # Higher strike -> larger put delta magnitude.
            if d < target:
                lo = mid
            else:
                hi = mid

    return round_to_increment(0.5 * (lo + hi), increment)


def symbol_of(market: MarketState) -> str:
    """
    Underlying symbol for contracts built against a market state.

    `MarketState` intentionally does not carry a symbol — it is a snapshot of
    prices and rates, not an instrument — so builders that need a label use this
    placeholder unless the caller names the underlying explicitly.
    """
    return getattr(market, "symbol", "UNDERLYING")


# --------------------------------------------------------------------------
# Directional and hedged structures
# --------------------------------------------------------------------------

def covered_call(
    underlying: str,
    spot_shares: float,
    call_strike: float,
    expiry: date,
    style: Style | str = Style.AMERICAN,
    multiplier: float = 100.0,
) -> Position:
    """
    Long stock, short an out-of-the-money call against it.

    Sells the upside above the strike in exchange for the premium: caps the
    payoff, funds a small buffer against a fall. The short call is where
    American exercise genuinely matters — an in-the-money short call can be
    assigned early, especially before a dividend, which is why `style` defaults
    to American and the backtester models assignment.
    """
    contracts = spot_shares / multiplier
    return Position(
        name=f"covered call {underlying} {call_strike:g}",
        legs=[
            Leg(Underlying(underlying), spot_shares),
            Leg(option(underlying, Right.CALL, call_strike, expiry, style, multiplier), -contracts),
        ],
    )


def protective_put(
    underlying: str,
    spot_shares: float,
    put_strike: float,
    expiry: date,
    style: Style | str = Style.AMERICAN,
    multiplier: float = 100.0,
) -> Position:
    """Long stock plus a long put: insurance with a deductible of (spot - strike)."""
    contracts = spot_shares / multiplier
    return Position(
        name=f"protective put {underlying} {put_strike:g}",
        legs=[
            Leg(Underlying(underlying), spot_shares),
            Leg(option(underlying, Right.PUT, put_strike, expiry, style, multiplier), contracts),
        ],
    )


def collar(
    underlying: str,
    spot_shares: float,
    put_strike: float,
    call_strike: float,
    expiry: date,
    style: Style | str = Style.AMERICAN,
    multiplier: float = 100.0,
) -> Position:
    """
    Long stock, long put, short call: a protective put paid for by selling
    upside. Often struck so the premiums roughly offset (a "zero-cost collar").
    """
    if put_strike >= call_strike:
        raise ValueError("collar needs put_strike < call_strike")
    contracts = spot_shares / multiplier
    return Position(
        name=f"collar {underlying} {put_strike:g}/{call_strike:g}",
        legs=[
            Leg(Underlying(underlying), spot_shares),
            Leg(option(underlying, Right.PUT, put_strike, expiry, style, multiplier), contracts),
            Leg(option(underlying, Right.CALL, call_strike, expiry, style, multiplier), -contracts),
        ],
    )


def vertical_spread(
    underlying: str,
    right: Right | str,
    long_strike: float,
    short_strike: float,
    expiry: date,
    quantity: float = 1.0,
    style: Style | str = Style.AMERICAN,
    multiplier: float = 100.0,
) -> Position:
    """
    Two same-expiry options of the same right, one long and one short.

    This single function covers all four verticals; which one you get follows
    from the strikes and the right:

    - call, long_strike < short_strike -> bull call spread (debit, bullish)
    - call, long_strike > short_strike -> bear call spread (credit, bearish)
    - put,  long_strike > short_strike -> bear put spread  (debit, bearish)
    - put,  long_strike < short_strike -> bull put spread  (credit, bullish)

    Risk is capped on both sides by construction, which is the entire reason
    verticals exist: the long leg bounds the loss the short leg can inflict.
    """
    if long_strike == short_strike:
        raise ValueError("vertical spread needs two distinct strikes")
    return Position(
        name=f"{Right(right).value} vertical {underlying} {long_strike:g}/{short_strike:g}",
        legs=[
            Leg(option(underlying, right, long_strike, expiry, style, multiplier), quantity),
            Leg(option(underlying, right, short_strike, expiry, style, multiplier), -quantity),
        ],
    )


# --------------------------------------------------------------------------
# Volatility structures
# --------------------------------------------------------------------------

def straddle(
    underlying: str,
    strike: float,
    expiry: date,
    quantity: float = 1.0,
    style: Style | str = Style.AMERICAN,
    multiplier: float = 100.0,
) -> Position:
    """
    A call and a put at the same strike.

    Long (quantity > 0) is a bet on movement in either direction and is long
    gamma and vega; short is a bet on stillness, collecting premium against
    unlimited risk. Delta near zero at the money, so the initial exposure is to
    volatility rather than direction.
    """
    return Position(
        name=f"{'long' if quantity > 0 else 'short'} straddle {underlying} {strike:g}",
        legs=[
            Leg(option(underlying, Right.CALL, strike, expiry, style, multiplier), quantity),
            Leg(option(underlying, Right.PUT, strike, expiry, style, multiplier), quantity),
        ],
    )


def strangle(
    underlying: str,
    put_strike: float,
    call_strike: float,
    expiry: date,
    quantity: float = 1.0,
    style: Style | str = Style.AMERICAN,
    multiplier: float = 100.0,
) -> Position:
    """
    Out-of-the-money call plus out-of-the-money put.

    Cheaper than a straddle and needs a larger move to pay off. Short strangles
    are the canonical premium-selling trade: a wide profit zone, positive theta,
    and a tail that can take far more than the credit received.
    """
    if put_strike >= call_strike:
        raise ValueError("strangle needs put_strike < call_strike")
    return Position(
        name=f"{'long' if quantity > 0 else 'short'} strangle {underlying} {put_strike:g}/{call_strike:g}",
        legs=[
            Leg(option(underlying, Right.PUT, put_strike, expiry, style, multiplier), quantity),
            Leg(option(underlying, Right.CALL, call_strike, expiry, style, multiplier), quantity),
        ],
    )


def iron_condor(
    underlying: str,
    long_put: float,
    short_put: float,
    short_call: float,
    long_call: float,
    expiry: date,
    quantity: float = 1.0,
    style: Style | str = Style.AMERICAN,
    multiplier: float = 100.0,
) -> Position:
    """
    A short strangle with protective wings: four legs, defined risk.

    Strikes must be ordered long_put < short_put < short_call < long_call. Sells
    the inner strangle and buys the outer wings, so the maximum loss is the wing
    width minus the credit, rather than unbounded. This is the structure the
    backtest's headline example trades, because it is the one where transaction
    costs, early assignment and skew all matter at once — four legs means four
    bid/ask crossings per entry, which is exactly the friction that quietly
    kills premium-selling strategies.
    """
    if not (long_put < short_put < short_call < long_call):
        raise ValueError("iron condor needs long_put < short_put < short_call < long_call")
    return Position(
        name=f"iron condor {underlying} {long_put:g}/{short_put:g}/{short_call:g}/{long_call:g}",
        legs=[
            Leg(option(underlying, Right.PUT, long_put, expiry, style, multiplier), quantity),
            Leg(option(underlying, Right.PUT, short_put, expiry, style, multiplier), -quantity),
            Leg(option(underlying, Right.CALL, short_call, expiry, style, multiplier), -quantity),
            Leg(option(underlying, Right.CALL, long_call, expiry, style, multiplier), quantity),
        ],
    )


def butterfly(
    underlying: str,
    right: Right | str,
    lower: float,
    body: float,
    upper: float,
    expiry: date,
    quantity: float = 1.0,
    style: Style | str = Style.AMERICAN,
    multiplier: float = 100.0,
) -> Position:
    """
    Long one lower strike, short two of the body, long one upper.

    A bet that the underlying pins near the body strike. Cheap, with a sharp
    peak — the payoff is a tent — and the classic example of a structure whose
    value comes almost entirely from gamma and time rather than direction.
    """
    if not (lower < body < upper):
        raise ValueError("butterfly needs lower < body < upper")
    return Position(
        name=f"{Right(right).value} butterfly {underlying} {lower:g}/{body:g}/{upper:g}",
        legs=[
            Leg(option(underlying, right, lower, expiry, style, multiplier), quantity),
            Leg(option(underlying, right, body, expiry, style, multiplier), -2 * quantity),
            Leg(option(underlying, right, upper, expiry, style, multiplier), quantity),
        ],
    )


def calendar_spread(
    underlying: str,
    right: Right | str,
    strike: float,
    near_expiry: date,
    far_expiry: date,
    quantity: float = 1.0,
    style: Style | str = Style.AMERICAN,
    multiplier: float = 100.0,
) -> Position:
    """
    Short the near-dated option, long the far-dated one at the same strike.

    Included partly to exercise the multi-expiry path deliberately: a calendar
    has no single expiry, so `Position.payoff_at_expiry` refuses it and the
    caller must use `portfolio.value_curve` to mark it on a date. That refusal
    is a feature — the near leg dying while the far leg still holds time value
    is precisely what the trade is about, and a naive single-expiry diagram
    would misrepresent it.
    """
    if near_expiry >= far_expiry:
        raise ValueError("calendar spread needs near_expiry < far_expiry")
    return Position(
        name=f"{Right(right).value} calendar {underlying} {strike:g}",
        legs=[
            Leg(option(underlying, right, strike, near_expiry, style, multiplier), -quantity),
            Leg(option(underlying, right, strike, far_expiry, style, multiplier), quantity),
        ],
    )


# --------------------------------------------------------------------------
# Delta-targeted builders (what the backtester calls)
# --------------------------------------------------------------------------

def short_strangle_by_delta(
    market: MarketState,
    underlying: str,
    expiry: date,
    target_delta: float = 0.20,
    quantity: float = 1.0,
    increment: float = 1.0,
    style: Style | str = Style.AMERICAN,
) -> Position:
    """Short strangle with both legs struck at a target delta magnitude."""
    put_k = strike_for_delta(market, target_delta, expiry, is_call=False, increment=increment)
    call_k = strike_for_delta(market, target_delta, expiry, is_call=True, increment=increment)
    return strangle(underlying, put_k, call_k, expiry, quantity=-abs(quantity), style=style)


def iron_condor_by_delta(
    market: MarketState,
    underlying: str,
    expiry: date,
    short_delta: float = 0.20,
    wing_width: float = 5.0,
    quantity: float = 1.0,
    increment: float = 1.0,
    style: Style | str = Style.AMERICAN,
) -> Position:
    """
    Iron condor with the short strikes at a target delta and wings a fixed
    distance further out.

    Fixed-width wings (rather than delta-targeted wings) keep the maximum loss
    constant in dollars across the backtest, which makes position sizing and
    the resulting risk statistics comparable over time.
    """
    short_put = strike_for_delta(market, short_delta, expiry, is_call=False, increment=increment)
    short_call = strike_for_delta(market, short_delta, expiry, is_call=True, increment=increment)
    long_put = round_to_increment(short_put - wing_width, increment)
    long_call = round_to_increment(short_call + wing_width, increment)
    return iron_condor(underlying, long_put, short_put, short_call, long_call, expiry,
                       quantity=abs(quantity), style=style)


def covered_call_by_delta(
    market: MarketState,
    underlying: str,
    expiry: date,
    target_delta: float = 0.30,
    shares: float = 100.0,
    increment: float = 1.0,
    style: Style | str = Style.AMERICAN,
) -> Position:
    """Covered call whose short strike sits at a target delta."""
    call_k = strike_for_delta(market, target_delta, expiry, is_call=True, increment=increment)
    return covered_call(underlying, shares, call_k, expiry, style=style)
