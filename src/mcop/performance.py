"""
Performance and risk statistics for a backtest result.

Reading these numbers correctly matters more than computing them, so each
metric below carries a note on how it lies.

The headline caveat: **there is no margin model**, so every return here is
computed against `initial_capital`, which is an arbitrary denominator. Selling
one iron condor against $100k of cash produces a tiny, flattering-looking
return series with an enormous Sharpe, because the denominator is mostly idle
cash that dampens volatility without dampening P&L. Return-based metrics are
therefore comparable *between runs in this framework* but are not claims about
what the strategy would return on deployed capital. Trade-level statistics
(win rate, profit factor, expectancy) do not have this problem and are the more
honest read.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np

from .backtest import BacktestResult

__all__ = [
    "PerformanceReport",
    "summarize",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "historical_var",
]

TRADING_DAYS = 252.0


def max_drawdown(equity: np.ndarray) -> tuple[float, int, int]:
    """
    Largest peak-to-trough decline as a fraction, with its start and end indices.

    Computed on the running maximum, so it captures the worst experience of an
    investor who bought at the worst moment — which is the number that actually
    determines whether a strategy survives contact with a real allocator.
    """
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return 0.0, 0, 0
    running_max = np.maximum.accumulate(eq)
    drawdowns = eq / running_max - 1.0
    end = int(np.argmin(drawdowns))
    start = int(np.argmax(eq[: end + 1])) if end > 0 else 0
    return float(drawdowns[end]), start, end


def sharpe_ratio(returns: np.ndarray, rf_annual: float = 0.0, periods: float = TRADING_DAYS) -> float:
    """
    Annualised Sharpe ratio.

    Note this assumes returns are roughly i.i.d. and roughly normal. Option
    selling violates both: returns are a long series of small gains punctuated
    by rare large losses, which is exactly the shape Sharpe flatters most. A
    high Sharpe on a short-premium strategy should be read as "has not met its
    tail yet", not as evidence of quality — which is why `summarize` reports
    skew, kurtosis and worst-day alongside it.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    excess = r - rf_annual / periods
    sd = float(np.std(excess, ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(np.mean(excess) / sd * math.sqrt(periods))


def sortino_ratio(returns: np.ndarray, rf_annual: float = 0.0, periods: float = TRADING_DAYS) -> float:
    """
    Sharpe variant penalising only downside deviation.

    More appropriate than Sharpe for asymmetric payoffs, since upside volatility
    is not a risk anyone objects to.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    excess = r - rf_annual / periods
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("inf")
    dd = float(np.sqrt(np.mean(np.square(downside))))
    if dd == 0.0:
        return float("nan")
    return float(np.mean(excess) / dd * math.sqrt(periods))


def historical_var(returns: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """
    Historical Value-at-Risk and Conditional VaR (expected shortfall).

    Empirical quantiles rather than a normal approximation, deliberately: a
    parametric VaR on a short-option strategy understates the tail precisely
    because the tail is the non-normal part. CVaR — the mean loss *given* a
    breach — is the more informative of the two and is what modern risk
    frameworks have largely moved to.

    Returns both as negative numbers (losses).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan"), float("nan")
    var = float(np.quantile(r, 1.0 - level))
    tail = r[r <= var]
    cvar = float(tail.mean()) if tail.size else var
    return var, cvar


@dataclass
class PerformanceReport:
    """Summary statistics for one backtest run."""

    symbol: str
    strategy: str
    start: str
    end: str
    years: float

    initial_capital: float
    final_equity: float
    total_return: float
    cagr: float

    ann_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float

    var_95: float
    cvar_95: float
    worst_day: float
    best_day: float
    return_skew: float
    return_kurtosis: float

    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    avg_days_held: float
    total_pnl: float
    total_commissions: float
    exit_reasons: dict

    underlying_return: float
    underlying_max_drawdown: float

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        pct = lambda x: f"{100 * x:,.2f}%"  # noqa: E731
        lines = [
            f"{self.strategy} on {self.symbol}   {self.start} to {self.end}  ({self.years:.2f}y)",
            "-" * 72,
            "RETURNS",
            f"  initial capital     {self.initial_capital:>14,.2f}",
            f"  final equity        {self.final_equity:>14,.2f}",
            f"  total return        {pct(self.total_return):>14}",
            f"  CAGR                {pct(self.cagr):>14}",
            "",
            "RISK",
            f"  annualised vol      {pct(self.ann_volatility):>14}",
            f"  Sharpe              {self.sharpe:>14.2f}",
            f"  Sortino             {self.sortino:>14.2f}",
            f"  max drawdown        {pct(self.max_drawdown):>14}",
            f"  Calmar              {self.calmar:>14.2f}",
            f"  daily VaR (95%)     {pct(self.var_95):>14}",
            f"  daily CVaR (95%)    {pct(self.cvar_95):>14}",
            f"  worst day           {pct(self.worst_day):>14}",
            f"  return skew         {self.return_skew:>14.2f}",
            f"  excess kurtosis     {self.return_kurtosis:>14.2f}",
            "",
            "TRADES",
            f"  closed trades       {self.n_trades:>14d}",
            f"  win rate            {pct(self.win_rate):>14}",
            f"  average win         {self.avg_win:>14,.2f}",
            f"  average loss        {self.avg_loss:>14,.2f}",
            f"  profit factor       {self.profit_factor:>14.2f}",
            f"  expectancy / trade  {self.expectancy:>14,.2f}",
            f"  avg days held       {self.avg_days_held:>14.1f}",
            f"  total P&L           {self.total_pnl:>14,.2f}",
            f"  commissions paid    {self.total_commissions:>14,.2f}",
            f"  exits               {dict(self.exit_reasons)}",
            "",
            "BENCHMARK (buy and hold underlying)",
            f"  underlying return   {pct(self.underlying_return):>14}",
            f"  underlying max DD   {pct(self.underlying_max_drawdown):>14}",
        ]
        return "\n".join(lines)


def summarize(result: BacktestResult, rf_annual: float = 0.0) -> PerformanceReport:
    """
    Compute the full statistics bundle for a backtest result.

    The underlying buy-and-hold comparison is included because it is the
    question any reader will ask first: an option strategy that underperforms
    simply holding the stock, on worse drawdowns, has not earned its complexity.

    Why the risk-free rate defaults to zero rather than to `config.r`
    -----------------------------------------------------------------
    The backtester discounts option prices at r but does **not** credit idle
    cash with interest. Charging the strategy a risk-free hurdle it was never
    given the chance to earn is inconsistent, and produces the actively
    misleading result of a negative Sharpe on a profitable strategy — which is
    what happens here, since most of the notional capital sits idle and the
    strategy's return on it is small.

    Sharpe is therefore computed on raw returns by default. Pass `rf_annual`
    explicitly if you model financing separately. Either way, see the module
    docstring: with no margin model the denominator is arbitrary, and the
    trade-level statistics are the more meaningful read.
    """
    eq = np.asarray(result.equity, dtype=float)
    rets = result.equity_returns()
    rets = rets[np.isfinite(rets)]

    rf = rf_annual

    n_days = len(result.dates)
    years = max((result.dates[-1] - result.dates[0]).days / 365.25, 1e-9)

    initial = float(result.config.initial_capital)
    final = float(eq[-1]) if eq.size else initial
    total_return = final / initial - 1.0
    cagr = (final / initial) ** (1.0 / years) - 1.0 if final > 0 else -1.0

    ann_vol = float(np.std(rets, ddof=1) * math.sqrt(TRADING_DAYS)) if rets.size > 1 else 0.0
    mdd, _, _ = max_drawdown(eq)
    calmar = (cagr / abs(mdd)) if mdd < 0 else float("inf")

    var95, cvar95 = historical_var(rets, 0.95)

    if rets.size > 2:
        centred = rets - rets.mean()
        sd = rets.std(ddof=0)
        skew = float(np.mean(centred**3) / sd**3) if sd > 0 else 0.0
        kurt = float(np.mean(centred**4) / sd**4 - 3.0) if sd > 0 else 0.0
    else:
        skew = kurt = 0.0

    closed = result.closed_trades
    pnls = np.array([t.pnl for t in closed], dtype=float) if closed else np.array([])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    gross_win = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(-losses.sum()) if losses.size else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    held = [t.days_held for t in closed if t.days_held is not None]

    underlying = np.asarray(result.underlying, dtype=float)
    u_ret = float(underlying[-1] / underlying[0] - 1.0) if underlying.size else 0.0
    u_mdd, _, _ = max_drawdown(underlying)

    return PerformanceReport(
        symbol=result.symbol,
        strategy=result.config.strategy,
        start=result.dates[0].isoformat(),
        end=result.dates[-1].isoformat(),
        years=years,
        initial_capital=initial,
        final_equity=final,
        total_return=total_return,
        cagr=cagr,
        ann_volatility=ann_vol,
        sharpe=sharpe_ratio(rets, rf),
        sortino=sortino_ratio(rets, rf),
        max_drawdown=mdd,
        calmar=calmar,
        var_95=var95,
        cvar_95=cvar95,
        worst_day=float(rets.min()) if rets.size else 0.0,
        best_day=float(rets.max()) if rets.size else 0.0,
        return_skew=skew,
        return_kurtosis=kurt,
        n_trades=len(closed),
        win_rate=float(wins.size / len(closed)) if closed else 0.0,
        avg_win=float(wins.mean()) if wins.size else 0.0,
        avg_loss=float(losses.mean()) if losses.size else 0.0,
        profit_factor=profit_factor,
        expectancy=float(pnls.mean()) if pnls.size else 0.0,
        avg_days_held=float(np.mean(held)) if held else 0.0,
        total_pnl=float(pnls.sum()) if pnls.size else 0.0,
        total_commissions=float(sum(t.commissions for t in closed)),
        exit_reasons=dict(Counter(t.reason for t in closed)),
        underlying_return=u_ret,
        underlying_max_drawdown=u_mdd,
    )
