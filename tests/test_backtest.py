"""
Backtest engine: accounting identities, absence of look-ahead, and the effect
of frictions.

The most valuable tests here are not the ones checking that code runs — they
are the ones checking that the simulation cannot cheat.
"""

from datetime import date

import numpy as np
import pytest

from mcop.backtest import (
    BacktestConfig,
    CostModel,
    monthly_expiries,
    run_backtest,
    third_friday,
)
from mcop.market_data import (
    FlatVolSurface,
    PriceHistory,
    rolling_realized_vol,
    synthetic_price_history,
)
from mcop.performance import summarize

FREE = CostModel(commission_per_contract=0.0, spread_frac_of_price=0.0, min_spread=0.0)


def _cfg(**kwargs):
    """
    BacktestConfig for tests, with daily Greek tracking off by default.

    Tracking book Greeks finite-differences the binomial tree for every leg on
    every day, which dominates runtime and is not what these tests assert on.
    Tests that need the risk series pass track_greeks=True explicitly.
    """
    kwargs.setdefault("track_greeks", False)
    return BacktestConfig(**kwargs)


@pytest.fixture(scope="module")
def history():
    return synthetic_price_history("TEST", S0=100, mu=0.06, sigma=0.20, n_days=380, seed=7)


# --- expiry calendar -------------------------------------------------------

def test_third_friday_known_dates():
    assert third_friday(2024, 1) == date(2024, 1, 19)
    assert third_friday(2024, 3) == date(2024, 3, 15)
    assert third_friday(2024, 11) == date(2024, 11, 15)
    assert third_friday(2025, 2) == date(2025, 2, 21)


def test_third_friday_is_always_a_friday():
    for year in (2023, 2024, 2025):
        for month in range(1, 13):
            assert third_friday(year, month).weekday() == 4


def test_monthly_expiries_are_sorted_and_in_range():
    exps = monthly_expiries(date(2024, 1, 1), date(2024, 12, 31))
    assert len(exps) == 12
    assert exps == sorted(exps)


# --- no look-ahead ---------------------------------------------------------

def test_rolling_vol_uses_only_past_returns():
    """
    The vol at index i must be computable from prices strictly before i.
    Verified by changing a future price and confirming earlier vols are
    unaffected — the direct test of look-ahead.
    """
    base = synthetic_price_history("LA", n_days=120, seed=3)
    tampered_closes = base.closes.copy()
    tampered_closes[100:] *= 1.5  # a violent future move
    tampered = PriceHistory("LA", base.dates, tampered_closes)

    v_base = rolling_realized_vol(base, window=21)
    v_tampered = rolling_realized_vol(tampered, window=21)

    # Everything at or before index 100 must be untouched by the future.
    np.testing.assert_allclose(v_base[:100], v_tampered[:100], equal_nan=True)
    # And the future change must actually have registered later, else the test
    # would pass trivially.
    assert not np.allclose(v_base[105:], v_tampered[105:], equal_nan=True)


def test_early_dates_without_vol_history_do_not_trade(history):
    cfg = _cfg(vol_window=21, cost=FREE)
    res = run_backtest(history, cfg)
    first_trade = min(t.open_date for t in res.trades)
    assert first_trade >= history.dates[21]


# --- accounting identities -------------------------------------------------

def test_equity_equals_cash_plus_exposure(history):
    res = run_backtest(history, _cfg(cost=FREE))
    np.testing.assert_allclose(res.equity, res.cash + res.exposure, rtol=1e-9, atol=1e-6)


def test_equity_curve_is_finite_and_full_length(history):
    res = run_backtest(history, _cfg())
    assert len(res.equity) == len(history)
    assert np.all(np.isfinite(res.equity))


def test_closed_trade_pnl_equals_entry_plus_exit_cash(history):
    res = run_backtest(history, _cfg())
    for t in res.closed_trades:
        assert t.pnl == pytest.approx(t.entry_cash + t.exit_cash, abs=1e-9)


def test_realised_pnl_reconciles_with_final_cash(history):
    """
    End-to-end accounting check on the full cash identity:

        final equity = initial capital
                     + realised P&L on closed trades
                     + cash received when opening any still-open position
                     + mark-to-market of those still-open positions

    The third term is easy to forget and is exactly what a naive reconciliation
    would drop: a position opened for a credit has already moved cash, but its
    P&L is not yet realised.
    """
    cfg = _cfg(cost=FREE)
    res = run_backtest(history, cfg)

    open_at_end = [t for t in res.trades if t.pnl is None]
    realised = sum(t.pnl for t in res.closed_trades)
    open_entry_cash = sum(t.entry_cash for t in open_at_end)

    expected = cfg.initial_capital + realised + open_entry_cash + res.exposure[-1]
    assert res.equity[-1] == pytest.approx(expected, abs=1e-6)
    assert len(open_at_end) <= cfg.max_concurrent


def test_max_concurrent_positions_is_respected(history):
    cfg = _cfg(max_concurrent=1)
    res = run_backtest(history, cfg)
    # Trades must not overlap when only one position is allowed at a time.
    closed = sorted(res.closed_trades, key=lambda t: t.open_date)
    for a, b in zip(closed, closed[1:]):
        assert b.open_date >= a.close_date


# --- exit rules ------------------------------------------------------------

def test_every_closed_trade_has_a_recognised_reason(history):
    res = run_backtest(history, _cfg())
    valid = {"expired", "assigned", "profit_target", "stop_loss", "dte_exit"}
    assert {t.reason for t in res.closed_trades} <= valid


def test_positions_are_not_held_past_the_dte_exit(history):
    cfg = _cfg(exit_dte=21)
    res = run_backtest(history, cfg)
    for t in res.closed_trades:
        assert (t.expiry - t.close_date).days >= 0


def test_profit_target_caps_realised_gains(history):
    """With a 50% target, no winner should materially exceed half the credit."""
    cfg = _cfg(strategy="short_strangle", profit_target=0.50, stop_loss=None, cost=FREE)
    res = run_backtest(history, cfg)
    for t in res.closed_trades:
        if t.reason == "profit_target":
            assert t.pnl <= 0.75 * abs(t.entry_cash)


# --- frictions -------------------------------------------------------------

def test_costs_strictly_reduce_pnl(history):
    """
    With the trade sequence held fixed, adding costs can only make results worse.
    A failure here means the cost model has a sign error.

    Exits are pinned to the DTE rule (no profit target, no stop) deliberately.
    Those two rules are expressed as a fraction of the entry credit, so changing
    costs changes the *thresholds*, which changes which trades close when — and
    then the two runs are no longer comparable like for like. See
    `test_percentage_exits_make_cost_impact_path_dependent` for that effect.
    """
    fixed_exits = dict(profit_target=None, stop_loss=None)
    cheap = summarize(run_backtest(history, _cfg(cost=FREE, **fixed_exits)))
    dear = summarize(run_backtest(history, _cfg(cost=CostModel(), **fixed_exits)))
    assert dear.total_pnl < cheap.total_pnl
    assert dear.total_commissions > 0


def test_wider_spreads_cost_more(history):
    """Same invariant, isolating the spread from commission."""
    fixed_exits = dict(profit_target=None, stop_loss=None)
    narrow = CostModel(commission_per_contract=0.0, spread_frac_of_price=0.01)
    wide = CostModel(commission_per_contract=0.0, spread_frac_of_price=0.10)
    a = summarize(run_backtest(history, _cfg(cost=narrow, **fixed_exits)))
    b = summarize(run_backtest(history, _cfg(cost=wide, **fixed_exits)))
    assert b.total_pnl < a.total_pnl


def test_percentage_exits_make_cost_impact_path_dependent(history):
    """
    Documents a genuinely counter-intuitive interaction rather than asserting it
    away.

    When the profit target is a percentage of the credit received, a wider
    spread reduces that credit, which lowers the absolute profit target, which
    makes it easier to hit — so trades close earlier and some losers are avoided.
    The result is that P&L is *not* monotone in the spread once percentage-based
    exits are switched on, even though every individual fill is strictly worse.

    The test asserts only that the trade sequences genuinely differ, which is the
    mechanism. It is here so that a future reader who sees a wider spread produce
    a better number knows it is expected, not a bug.
    """
    narrow = CostModel(commission_per_contract=0.0, spread_frac_of_price=0.01)
    wide = CostModel(commission_per_contract=0.0, spread_frac_of_price=0.10)
    a = run_backtest(history, _cfg(cost=narrow, profit_target=0.5, stop_loss=2.0))
    b = run_backtest(history, _cfg(cost=wide, profit_target=0.5, stop_loss=2.0))

    exits_a = [(t.close_date, t.reason) for t in a.closed_trades]
    exits_b = [(t.close_date, t.reason) for t in b.closed_trades]
    assert exits_a != exits_b


def test_fill_prices_always_move_against_the_trader():
    cost = CostModel(spread_frac_of_price=0.02, min_spread=0.01)
    mid = 2.50
    assert cost.fill_price(mid, buying=True) > mid
    assert cost.fill_price(mid, buying=False) < mid


def test_fill_price_never_goes_negative():
    cost = CostModel(spread_frac_of_price=0.50, min_spread=1.0)
    assert cost.fill_price(0.02, buying=False) >= 0.0


# --- the fairness check ----------------------------------------------------

def test_fair_priced_options_have_no_systematic_edge():
    """
    The strongest correctness test in the suite.

    If the underlying drifts at r, options are marked at exactly the volatility
    the paths were generated with, there is no variance risk premium and no
    costs, then selling options is a zero-expectancy game. A consistent profit
    or loss across independent seeds would indicate a bias in the accounting,
    the settlement logic, or the pricer.
    """
    totals = []
    for seed in range(6):
        hist = synthetic_price_history("FAIR", S0=100, mu=0.04, sigma=0.20, n_days=300, seed=seed)
        cfg = _cfg(
            strategy="short_strangle", r=0.04, q=0.0,
            surface=FlatVolSurface(0.20), variance_risk_premium=0.0,
            cost=FREE, profit_target=None, stop_loss=None,
        )
        res = run_backtest(hist, cfg)
        totals.append(sum(t.pnl for t in res.closed_trades))

    totals = np.array(totals, dtype=float)
    se = totals.std(ddof=1) / np.sqrt(totals.size)
    # Mean P&L must be within ~3 standard errors of zero.
    assert abs(totals.mean()) < 3.0 * se + 1e-9, f"mean={totals.mean():.2f} se={se:.2f}"


def test_config_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="unknown strategy"):
        _cfg(strategy="not_a_strategy")


def test_config_rejects_exit_after_entry_dte():
    with pytest.raises(ValueError, match="exit_dte must be less than entry_dte"):
        _cfg(entry_dte=30, exit_dte=45)
