"""
Walk-forward backtest engine for option strategies.

What this does, in one paragraph: walk a real underlying price history forward
one trading day at a time; on each day, re-anchor a volatility surface to
trailing realised vol, mark every open position, apply exit rules, settle
anything that expired, and open a new position when the rules allow. Cash and
mark-to-market are tracked separately so the equity curve is auditable.

The three things that make a backtest honest
---------------------------------------------

**1. No look-ahead.** Every decision on date `i` uses information available
strictly before the close of `i`. Realised volatility at `i` is computed from
returns up to `i-1` (see `market_data.rolling_realized_vol`), and the engine
never indexes forward. This is the single easiest way to produce a spectacular
and entirely fake equity curve, so the constraint is enforced structurally
rather than by care.

**2. Transaction costs that reflect options, not equities.** Option bid/ask
spreads are wide — often 1-5% of premium on liquid names, far worse on
anything else — and a four-legged iron condor crosses that spread four times on
entry and four more on exit. Backtests that assume mid-price fills turn
losing premium-selling strategies into winners; this is probably the most
common way an options backtest lies. `CostModel` charges a half-spread on every
leg in both directions plus per-contract commission, and
`benchmarks/bench_backtest.py` shows the same strategy with and without costs
so the size of the effect is visible rather than assumed away.

**3. Early assignment.** Short American options can be assigned before expiry.
The engine flags a short leg as assigned when its extrinsic value collapses
(the rational-exercise condition) and closes the position, because a real
assignment converts an option position into a stock position overnight and ends
the trade as designed.

Known simplifications, stated rather than hidden
-------------------------------------------------
- Option prices are **modelled, not observed** — see the `market_data` module
  docstring. This is the dominant caveat on every number produced here.
- Assignment closes the whole position at that day's marks. Cash-settling an
  assigned leg at intrinsic and liquidating the resulting stock is P&L-
  equivalent to physical delivery, but the follow-on position an iron condor
  would leave is not modelled.
- No margin model. Position sizing is by contract count, so the equity curve
  does not reflect the buying power a short-premium strategy actually consumes,
  and returns are therefore not capital-efficiency-adjusted.
- Fills are always available at the modelled price. Real illiquidity means
  sometimes there is no fill at any sane price, particularly in the stress
  scenarios where an exit matters most.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from .instruments import Leg, OptionContract, Position, Underlying
from .market_data import (
    MarketState,
    PriceHistory,
    SkewedVolSurface,
    VolSurface,
    rolling_realized_vol,
)
from .portfolio import CRR_STEPS, position_value, position_greeks, price_instrument
from .strategies import (
    covered_call_by_delta,
    iron_condor_by_delta,
    short_strangle_by_delta,
    strike_for_delta,
    option,
)
from .instruments import Right, Style

__all__ = [
    "CostModel",
    "BacktestConfig",
    "Trade",
    "BacktestResult",
    "third_friday",
    "monthly_expiries",
    "run_backtest",
    "STRATEGY_BUILDERS",
]


# --------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------

@dataclass
class CostModel:
    """
    Transaction costs for option and stock legs.

    The spread is modelled as a fraction of the option's mid price with an
    absolute floor, because that is how option quotes actually behave: a $0.20
    option does not have a $0.20 spread, but nor does it have a $0.001 one.
    Crossing costs `spread_cross` of that width — 0.5 means paying the full
    half-spread, i.e. taking the offer, which is what a market order does.

    Defaults are deliberately mid-range retail: $0.65/contract commission and a
    2% spread. Set `spread_frac_of_price=0.0` and `commission_per_contract=0.0`
    to measure the frictionless case and see how much of a strategy's edge is
    real.
    """

    commission_per_contract: float = 0.65
    commission_per_share: float = 0.0
    spread_frac_of_price: float = 0.02
    min_spread: float = 0.01
    spread_cross: float = 0.5

    def half_spread(self, mid: float) -> float:
        width = max(self.spread_frac_of_price * abs(mid), self.min_spread)
        return self.spread_cross * width

    def fill_price(self, mid: float, buying: bool, is_option: bool = True) -> float:
        """
        Price actually transacted at.

        Buying pays up, selling receives less — the sign of the adjustment is
        always against the trader, which is the whole point. Floored at zero so
        a deep out-of-the-money option cannot be sold for a negative price.
        """
        if not is_option:
            return float(mid)  # stock spreads are negligible at this granularity
        adj = self.half_spread(mid)
        return float(max(mid + adj, 0.0) if buying else max(mid - adj, 0.0))

    def commission(self, position: Position) -> float:
        total = 0.0
        for leg in position.legs:
            if leg.is_option:
                total += self.commission_per_contract * abs(leg.quantity)
            else:
                total += self.commission_per_share * abs(leg.quantity)
        return float(total)


# --------------------------------------------------------------------------
# Expiry calendar
# --------------------------------------------------------------------------

def third_friday(year: int, month: int) -> date:
    """
    Third Friday of the month — the standard US monthly option expiry.

    Using real listed expiries rather than "today + 45 days" matters: it means
    days-to-expiry at entry drifts between roughly 30 and 60 across the
    backtest, exactly as it would for a trader rolling monthly, instead of
    being pinned at a constant that no real chain would offer.
    """
    d = date(year, month, 1)
    # weekday(): Monday=0 ... Friday=4
    offset = (4 - d.weekday()) % 7
    return d + timedelta(days=offset + 14)


def monthly_expiries(start: date, end: date) -> list[date]:
    """All third Fridays in [start, end]."""
    out: list[date] = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        exp = third_friday(y, m)
        if start <= exp <= end:
            out.append(exp)
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _select_expiry(asof: date, target_dte: int, expiries: list[date], min_dte: int = 7) -> date | None:
    """Listed expiry whose DTE is closest to the target, ignoring anything too near."""
    candidates = [e for e in expiries if (e - asof).days >= min_dte]
    if not candidates:
        return None
    return min(candidates, key=lambda e: abs((e - asof).days - target_dte))


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def _short_put_by_delta(market, underlying, expiry, target_delta=0.20, quantity=1.0,
                        increment=1.0, style=Style.AMERICAN) -> Position:
    """Cash-secured short put at a target delta."""
    k = strike_for_delta(market, target_delta, expiry, is_call=False, increment=increment)
    return Position(
        name=f"short put {underlying} {k:g}",
        legs=[Leg(option(underlying, Right.PUT, k, expiry, style), -abs(quantity))],
    )


STRATEGY_BUILDERS = {
    "iron_condor": lambda mkt, sym, exp, cfg: iron_condor_by_delta(
        mkt, sym, exp, short_delta=cfg.short_delta, wing_width=cfg.wing_width,
        quantity=cfg.contracts, increment=cfg.strike_increment,
    ),
    "short_strangle": lambda mkt, sym, exp, cfg: short_strangle_by_delta(
        mkt, sym, exp, target_delta=cfg.short_delta,
        quantity=cfg.contracts, increment=cfg.strike_increment,
    ),
    "short_put": lambda mkt, sym, exp, cfg: _short_put_by_delta(
        mkt, sym, exp, target_delta=cfg.short_delta,
        quantity=cfg.contracts, increment=cfg.strike_increment,
    ),
    "covered_call": lambda mkt, sym, exp, cfg: covered_call_by_delta(
        mkt, sym, exp, target_delta=cfg.short_delta,
        shares=100.0 * cfg.contracts, increment=cfg.strike_increment,
    ),
}


@dataclass
class BacktestConfig:
    """
    Every knob the engine reads, in one place.

    Defaults describe a conventional monthly premium-selling programme: enter
    ~45 days to expiry, take profit at half the credit, cut at twice the credit,
    and never hold into the final three weeks where gamma risk accelerates
    sharply.
    """

    strategy: str = "iron_condor"
    initial_capital: float = 100_000.0

    # Entry
    entry_dte: int = 45
    short_delta: float = 0.20
    wing_width: float = 5.0
    contracts: float = 1.0
    strike_increment: float = 1.0
    max_concurrent: int = 1

    # Exit
    exit_dte: int = 21
    profit_target: float | None = 0.50   # fraction of |entry cash| captured
    stop_loss: float | None = 2.00       # multiple of |entry cash| lost

    # Market model
    r: float = 0.04
    q: float = 0.0
    vol_window: int = 21
    variance_risk_premium: float = 0.15  # implied = realised * (1 + vrp)
    vol_floor: float = 0.05
    surface: VolSurface | None = None    # shape template; level is re-anchored daily

    # Frictions
    cost: CostModel = field(default_factory=CostModel)
    assignment_extrinsic_threshold: float = 0.05
    model_early_assignment: bool = True

    crr_steps: int = CRR_STEPS

    # Recording the book's net delta each day means finite-differencing the tree
    # for every leg — roughly nine extra valuations per leg per day, which
    # dominates runtime. Leave it on when you want the risk series; turn it off
    # when only the equity curve and trade log matter.
    track_greeks: bool = True

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGY_BUILDERS:
            raise ValueError(
                f"unknown strategy {self.strategy!r}; choose from {sorted(STRATEGY_BUILDERS)}"
            )
        if self.exit_dte >= self.entry_dte:
            raise ValueError("exit_dte must be less than entry_dte")


# --------------------------------------------------------------------------
# Trade records
# --------------------------------------------------------------------------

@dataclass
class Trade:
    """One round trip, with enough detail to audit any single result."""

    strategy: str
    open_date: date
    close_date: date | None
    expiry: date
    entry_spot: float
    exit_spot: float | None
    entry_cash: float           # signed cash at open: positive = credit received
    exit_cash: float | None     # signed cash at close
    commissions: float
    pnl: float | None
    reason: str | None
    legs: str

    @property
    def is_credit(self) -> bool:
        return self.entry_cash > 0

    @property
    def days_held(self) -> int | None:
        if self.close_date is None:
            return None
        return (self.close_date - self.open_date).days


@dataclass
class _OpenPosition:
    """Engine-internal state for a live position."""

    position: Position
    trade: Trade
    entry_cash: float


@dataclass
class BacktestResult:
    """
    Everything the run produced.

    `equity` is cash plus mark-to-market of open positions, so it is a genuine
    account value on every date, not a sum of closed-trade P&L.
    """

    dates: list[date]
    equity: np.ndarray
    cash: np.ndarray
    exposure: np.ndarray       # MTM of open positions
    net_delta: np.ndarray      # share-equivalent delta of the book
    trades: list[Trade]
    config: BacktestConfig
    symbol: str
    underlying: np.ndarray

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.pnl is not None]

    def equity_returns(self) -> np.ndarray:
        """Daily simple returns of the equity curve."""
        eq = np.asarray(self.equity, dtype=float)
        return np.diff(eq) / eq[:-1]


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

def _open_cash_flow(position: Position, market: MarketState, cost: CostModel, crr_steps: int) -> float:
    """
    Signed cash from opening a position, after slippage and commission.

    Positive means a net credit was received. Each leg fills at the price that
    is worse for the trader: buys lift the offer, sells hit the bid.
    """
    total = 0.0
    for leg in position.legs:
        mid = price_instrument(leg.instrument, market, crr_steps)
        buying = leg.quantity > 0
        fill = cost.fill_price(mid, buying=buying, is_option=leg.is_option)
        # Acquiring exposure costs signed_shares * fill.
        total -= leg.signed_shares * fill
    return float(total - cost.commission(position))


def _close_cash_flow(position: Position, market: MarketState, cost: CostModel, crr_steps: int) -> float:
    """Signed cash from closing a position, after slippage and commission."""
    total = 0.0
    for leg in position.legs:
        mid = price_instrument(leg.instrument, market, crr_steps)
        # Closing reverses the trade: a long leg is sold, a short leg is bought back.
        buying = leg.quantity < 0
        fill = cost.fill_price(mid, buying=buying, is_option=leg.is_option)
        total += leg.signed_shares * fill
    return float(total - cost.commission(position))


def _settle_at_expiry(position: Position, spot: float) -> float:
    """
    Cash from settling every leg at intrinsic value on the expiry date.

    No slippage: expiring options settle at intrinsic against the settlement
    price, they are not traded out. Stock legs are liquidated at spot, which is
    what makes assignment P&L-equivalent to physical delivery (delivering shares
    at the strike and selling shares at spot while paying intrinsic on the
    option come to the same number).
    """
    total = 0.0
    for leg in position.legs:
        if isinstance(leg.instrument, Underlying):
            value = spot
        else:
            value = leg.instrument.intrinsic(spot)
        total += leg.signed_shares * value
    return float(total)


def _assignment_triggered(position: Position, market: MarketState, threshold: float, crr_steps: int) -> bool:
    """
    Would any short American leg plausibly be assigned today?

    The rational-exercise condition: a short in-the-money option is at risk once
    its **extrinsic value** (mid minus intrinsic) falls below a threshold,
    because the holder gives up nothing by exercising early. Deep ITM short
    options with days left are the realistic assignment case, and ignoring it
    flatters any short-premium strategy that lets losers run.
    """
    for leg in position.legs:
        if not leg.is_option or leg.quantity >= 0:
            continue
        contract: OptionContract = leg.instrument
        if contract.style is not Style.AMERICAN:
            continue
        intrinsic = contract.intrinsic(market.spot)
        if intrinsic <= 0:
            continue
        mid = price_instrument(contract, market, crr_steps)
        if (mid - intrinsic) < threshold:
            return True
    return False


def run_backtest(
    history: PriceHistory,
    config: BacktestConfig | None = None,
    verbose: bool = False,
) -> BacktestResult:
    """
    Run a walk-forward option strategy backtest over a price history.

    Parameters
    ----------
    history : PriceHistory
        Real or synthetic daily closes for the underlying.
    config : BacktestConfig
        Strategy, entry/exit rules, market model and cost assumptions.

    Returns
    -------
    BacktestResult
        Equity curve, cash, exposure, net delta, and the full trade log.
    """
    cfg = config or BacktestConfig()
    surface_template = cfg.surface or SkewedVolSurface()
    builder = STRATEGY_BUILDERS[cfg.strategy]

    # Trailing realised vol, aligned so index i uses only returns before i.
    rv = rolling_realized_vol(history, window=cfg.vol_window)
    expiries = monthly_expiries(history.dates[0], history.dates[-1] + timedelta(days=400))

    cash = cfg.initial_capital
    open_positions: list[_OpenPosition] = []
    trades: list[Trade] = []

    equity_curve, cash_curve, exposure_curve, delta_curve = [], [], [], []

    for i, today in enumerate(history.dates):
        spot = float(history.closes[i])

        # --- today's market state -------------------------------------------------
        realised = rv[i]
        if np.isnan(realised):
            # Not enough history to estimate vol: record a flat equity point and
            # trade nothing. Guessing a vol here would be a look-ahead in
            # disguise.
            equity_curve.append(cash)
            cash_curve.append(cash)
            exposure_curve.append(0.0)
            delta_curve.append(0.0)
            continue

        atm = max(realised * (1.0 + cfg.variance_risk_premium), cfg.vol_floor)
        surface = (
            surface_template.with_atm(atm)
            if isinstance(surface_template, SkewedVolSurface)
            else surface_template
        )
        market = MarketState(asof=today, spot=spot, r=cfg.r, q=cfg.q, vol_surface=surface)

        # --- manage open positions ------------------------------------------------
        still_open: list[_OpenPosition] = []
        for op in open_positions:
            pos, trade = op.position, op.trade
            expiry = min(pos.expiries)
            dte = (expiry - today).days

            reason: str | None = None
            settle_cash: float | None = None

            if dte <= 0:
                reason = "expired"
                settle_cash = _settle_at_expiry(pos, spot)
            else:
                mtm = position_value(pos, market, cfg.crr_steps)
                # P&L if closed at mid right now, before exit slippage.
                unrealised = op.entry_cash + mtm
                reference = abs(op.entry_cash) if op.entry_cash != 0 else 1.0

                if cfg.model_early_assignment and _assignment_triggered(
                    pos, market, cfg.assignment_extrinsic_threshold, cfg.crr_steps
                ):
                    reason = "assigned"
                elif cfg.profit_target is not None and unrealised >= cfg.profit_target * reference:
                    reason = "profit_target"
                elif cfg.stop_loss is not None and unrealised <= -cfg.stop_loss * reference:
                    reason = "stop_loss"
                elif dte <= cfg.exit_dte:
                    reason = "dte_exit"

                if reason is not None:
                    settle_cash = _close_cash_flow(pos, market, cfg.cost, cfg.crr_steps)

            if reason is None:
                still_open.append(op)
                continue

            cash += settle_cash
            trade.close_date = today
            trade.exit_spot = spot
            trade.exit_cash = settle_cash
            trade.pnl = op.entry_cash + settle_cash
            trade.reason = reason
            if verbose:
                print(f"{today} CLOSE {trade.strategy:14s} {reason:14s} pnl={trade.pnl:9.2f}")

        open_positions = still_open

        # --- consider a new entry -------------------------------------------------
        if len(open_positions) < cfg.max_concurrent:
            expiry = _select_expiry(today, cfg.entry_dte, expiries, min_dte=cfg.exit_dte + 1)
            if expiry is not None:
                try:
                    pos = builder(market, history.symbol, expiry, cfg)
                    entry_cash = _open_cash_flow(pos, market, cfg.cost, cfg.crr_steps)
                except (ValueError, KeyError) as exc:
                    # A strike solve can fail in extreme vol regimes; skipping the
                    # entry is the right response, not aborting the run.
                    if verbose:
                        print(f"{today} SKIP entry: {exc}")
                    pos = None
                    entry_cash = 0.0

                if pos is not None:
                    cash += entry_cash
                    trade = Trade(
                        strategy=cfg.strategy,
                        open_date=today,
                        close_date=None,
                        expiry=expiry,
                        entry_spot=spot,
                        exit_spot=None,
                        entry_cash=entry_cash,
                        exit_cash=None,
                        commissions=cfg.cost.commission(pos),
                        pnl=None,
                        reason=None,
                        legs=str(pos),
                    )
                    trades.append(trade)
                    open_positions.append(_OpenPosition(pos, trade, entry_cash))
                    if verbose:
                        print(f"{today} OPEN  {cfg.strategy:14s} exp={expiry} cash={entry_cash:9.2f}")

        # --- record the day -------------------------------------------------------
        exposure = sum(position_value(op.position, market, cfg.crr_steps) for op in open_positions)
        book_delta = (
            sum(position_greeks(op.position, market, cfg.crr_steps).delta for op in open_positions)
            if cfg.track_greeks
            else float("nan")
        )
        equity_curve.append(cash + exposure)
        cash_curve.append(cash)
        exposure_curve.append(exposure)
        delta_curve.append(book_delta)

    return BacktestResult(
        dates=list(history.dates),
        equity=np.array(equity_curve, dtype=float),
        cash=np.array(cash_curve, dtype=float),
        exposure=np.array(exposure_curve, dtype=float),
        net_delta=np.array(delta_curve, dtype=float),
        trades=trades,
        config=cfg,
        symbol=history.symbol,
        underlying=np.asarray(history.closes, dtype=float),
    )
