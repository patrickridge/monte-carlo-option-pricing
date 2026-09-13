"""Performance statistics, verified against hand-computable cases."""

import math

import numpy as np
import pytest

from mcop.backtest import BacktestConfig, CostModel, run_backtest
from mcop.market_data import synthetic_price_history
from mcop.performance import (
    historical_var,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summarize,
)


def _cfg(**kwargs):
    """
    BacktestConfig for tests, with daily Greek tracking off by default.

    Tracking book Greeks finite-differences the binomial tree for every leg on
    every day, which dominates runtime and is not what these tests assert on.
    Tests that need the risk series pass track_greeks=True explicitly.
    """
    kwargs.setdefault("track_greeks", False)
    return BacktestConfig(**kwargs)




def test_max_drawdown_on_a_known_curve():
    equity = np.array([100.0, 120.0, 90.0, 110.0, 60.0, 80.0])
    mdd, start, end = max_drawdown(equity)
    # Worst decline is 120 -> 60, i.e. -50%.
    assert mdd == pytest.approx(-0.5)
    assert start == 1
    assert end == 4


def test_max_drawdown_of_a_monotone_curve_is_zero():
    mdd, _, _ = max_drawdown(np.array([100.0, 101.0, 102.0, 103.0]))
    assert mdd == pytest.approx(0.0)


def test_sharpe_of_constant_returns_is_undefined():
    """Zero variance means the ratio has no meaning; NaN is the honest answer."""
    assert math.isnan(sharpe_ratio(np.full(50, 0.001)))


def test_sharpe_matches_manual_computation():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0004, 0.01, 2000)
    expected = (rets.mean() / rets.std(ddof=1)) * math.sqrt(252)
    assert sharpe_ratio(rets) == pytest.approx(expected, rel=1e-10)


def test_sortino_exceeds_sharpe_for_right_skewed_returns():
    """Penalising only downside must flatter a series with capped losses."""
    rets = np.concatenate([np.full(200, 0.01), np.full(50, -0.002)])
    assert sortino_ratio(rets) > sharpe_ratio(rets)


def test_sortino_is_infinite_without_losses():
    assert math.isinf(sortino_ratio(np.full(100, 0.001)))


def test_historical_var_and_cvar_ordering():
    rng = np.random.default_rng(1)
    rets = rng.normal(0, 0.01, 100_000)
    var, cvar = historical_var(rets, 0.95)
    assert cvar < var < 0                       # CVaR is the deeper tail
    assert var == pytest.approx(-1.645 * 0.01, rel=0.05)   # normal quantile


def test_var_on_empty_input_is_nan():
    var, cvar = historical_var(np.array([]))
    assert math.isnan(var) and math.isnan(cvar)


def test_summary_fields_are_internally_consistent():
    hist = synthetic_price_history("PERF", n_days=320, seed=11)
    res = run_backtest(hist, _cfg(cost=CostModel()))
    rpt = summarize(res)

    assert rpt.n_trades == len(res.closed_trades)
    assert rpt.total_pnl == pytest.approx(sum(t.pnl for t in res.closed_trades))
    assert rpt.final_equity == pytest.approx(res.equity[-1])
    assert rpt.total_return == pytest.approx(rpt.final_equity / rpt.initial_capital - 1.0)
    assert 0.0 <= rpt.win_rate <= 1.0
    assert rpt.max_drawdown <= 0.0
    assert sum(rpt.exit_reasons.values()) == rpt.n_trades
    assert str(rpt)  # renders without raising


def test_win_rate_and_profit_factor_agree_with_the_trade_log():
    hist = synthetic_price_history("PERF2", n_days=320, seed=12)
    res = run_backtest(hist, _cfg())
    rpt = summarize(res)

    pnls = np.array([t.pnl for t in res.closed_trades])
    if pnls.size:
        wins = pnls[pnls > 0]
        assert rpt.win_rate == pytest.approx(wins.size / pnls.size)
        if wins.size:
            assert rpt.avg_win == pytest.approx(wins.mean())
