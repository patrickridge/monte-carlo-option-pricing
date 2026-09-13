"""Valuation and risk netting, cross-checked against the closed form and MC."""

from datetime import date

import numpy as np
import pytest

from mcop.black_scholes import bs_price, bs_greeks
from mcop.instruments import Leg, OptionContract, Position, Right, Style, Underlying
from mcop.market_data import FlatVolSurface, MarketState
from mcop.portfolio import (
    breakevens,
    leg_greeks,
    leg_value,
    mc_price_leg,
    position_greeks,
    position_value,
    price_instrument,
    value_curve,
)
from mcop.strategies import iron_condor, straddle

ASOF = date(2024, 1, 15)
EXPIRY = date(2024, 4, 15)
VOL = 0.25


@pytest.fixture
def market():
    return MarketState(asof=ASOF, spot=100.0, r=0.04, q=0.0, vol_surface=FlatVolSurface(VOL))


def euro(strike, right=Right.CALL):
    return OptionContract("XYZ", right, strike, EXPIRY, Style.EUROPEAN, 100.0)


def amer(strike, right=Right.PUT):
    return OptionContract("XYZ", right, strike, EXPIRY, Style.AMERICAN, 100.0)


def test_european_leg_marks_to_black_scholes(market):
    contract = euro(100.0)
    T = contract.time_to_expiry(ASOF)
    expected = bs_price(100.0, 100.0, 0.04, 0.0, VOL, T, True)
    assert price_instrument(contract, market) == pytest.approx(expected, rel=1e-12)


def test_leg_value_includes_multiplier_and_sign(market):
    contract = euro(100.0)
    unit = price_instrument(contract, market)
    assert leg_value(Leg(contract, 2), market) == pytest.approx(200 * unit)
    assert leg_value(Leg(contract, -2), market) == pytest.approx(-200 * unit)


def test_american_put_is_worth_at_least_european(market):
    """Early exercise is an extra right; it cannot reduce value."""
    e = price_instrument(euro(110.0, Right.PUT), market)
    a = price_instrument(amer(110.0, Right.PUT), market)
    assert a >= e - 1e-9


def test_underlying_leg_is_pure_delta(market):
    g = leg_greeks(Leg(Underlying("XYZ"), 100), market)
    assert g.delta == pytest.approx(100.0)
    assert g.gamma == 0.0
    assert g.vega == 0.0
    assert g.theta == 0.0


def test_expired_leg_is_worth_intrinsic():
    market = MarketState(asof=date(2024, 5, 1), spot=120.0, r=0.04,
                         vol_surface=FlatVolSurface(VOL))
    assert price_instrument(euro(100.0), market) == pytest.approx(20.0)


def test_position_greeks_equal_sum_of_leg_greeks(market):
    pos = iron_condor("XYZ", 85, 90, 110, 115, EXPIRY, style=Style.EUROPEAN)
    net = position_greeks(pos, market)
    for field in ("delta", "gamma", "vega", "theta", "rho"):
        total = sum(getattr(leg_greeks(leg, market), field) for leg in pos.legs)
        assert getattr(net, field) == pytest.approx(total, rel=1e-10)


def test_short_option_has_negative_gamma_and_positive_theta(market):
    pos = straddle("XYZ", 100.0, EXPIRY, quantity=-1, style=Style.EUROPEAN)
    g = position_greeks(pos, market)
    assert g.gamma < 0
    assert g.theta > 0
    assert g.vega < 0


def test_atm_straddle_is_roughly_delta_neutral(market):
    g = position_greeks(straddle("XYZ", 100.0, EXPIRY, style=Style.EUROPEAN), market)
    assert abs(g.delta) < 15.0  # on 200 share-equivalents


def test_iron_condor_max_loss_equals_wing_width_minus_credit(market):
    """The defining property of the structure — verify it numerically."""
    pos = iron_condor("XYZ", 85, 90, 110, 115, EXPIRY, style=Style.EUROPEAN)
    entry = position_value(pos, market)  # negative => credit received
    spots = np.linspace(50, 160, 2000)
    pnl = pos.payoff_at_expiry(spots) - entry

    credit = -entry
    wing_width = (90 - 85) * 100
    assert pnl.max() == pytest.approx(credit, rel=1e-6)
    assert pnl.min() == pytest.approx(credit - wing_width, rel=1e-3)


def test_breakevens_sit_at_short_strikes_offset_by_credit(market):
    pos = iron_condor("XYZ", 85, 90, 110, 115, EXPIRY, style=Style.EUROPEAN)
    entry = position_value(pos, market)
    credit_per_share = -entry / 100.0

    be = breakevens(pos, entry_cost=entry)
    assert len(be) == 2
    assert be[0] == pytest.approx(90 - credit_per_share, abs=0.05)
    assert be[1] == pytest.approx(110 + credit_per_share, abs=0.05)


def test_value_curve_is_monotone_for_a_long_call(market):
    pos = Position(legs=[Leg(euro(100.0), 1)], name="long call")
    curve = value_curve(pos, market, np.linspace(70, 130, 25))
    assert np.all(np.diff(curve) > 0)


def test_value_curve_handles_multi_expiry_calendar(market):
    """The general path that payoff_at_expiry deliberately refuses."""
    pos = Position(
        name="calendar",
        legs=[
            Leg(OptionContract("XYZ", Right.CALL, 100, EXPIRY, Style.EUROPEAN), -1),
            Leg(OptionContract("XYZ", Right.CALL, 100, date(2024, 9, 20), Style.EUROPEAN), 1),
        ],
    )
    curve = value_curve(pos, market, np.array([90.0, 100.0, 110.0]))
    assert np.all(np.isfinite(curve))
    assert np.all(curve > 0)  # long the more valuable option


def test_monte_carlo_mark_agrees_with_analytic_mark(market):
    """
    The fast marks used in the backtest must agree with the Monte Carlo engine.
    This is the check that justifies swapping MC out for speed and determinism.
    """
    leg = Leg(euro(105.0), 1)
    analytic = leg_value(leg, market)
    mc = mc_price_leg(leg, market, n_paths=200_000, n_steps=50, seed=3)
    assert mc == pytest.approx(analytic, rel=0.01)


def test_binomial_tree_mark_agrees_with_lsm(market):
    """American marks: CRR tree vs Longstaff-Schwartz, within MC error."""
    leg = Leg(amer(105.0, Right.PUT), 1)
    tree = leg_value(leg, market)
    lsm = mc_price_leg(leg, market, n_paths=200_000, n_steps=100, seed=3)
    assert tree == pytest.approx(lsm, rel=0.02)
