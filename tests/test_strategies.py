"""Strategy builders: structure, sign conventions, and delta-targeted strikes."""

from datetime import date

import numpy as np
import pytest

from mcop.black_scholes import bs_greeks
from mcop.instruments import Right, Style
from mcop.market_data import FlatVolSurface, MarketState, SkewedVolSurface
from mcop.portfolio import position_greeks, position_value
from mcop.strategies import (
    butterfly,
    calendar_spread,
    collar,
    covered_call,
    iron_condor,
    iron_condor_by_delta,
    protective_put,
    round_to_increment,
    short_strangle_by_delta,
    straddle,
    strangle,
    strike_for_delta,
    vertical_spread,
)

ASOF = date(2024, 1, 15)
EXPIRY = date(2024, 3, 15)


@pytest.fixture
def market():
    return MarketState(asof=ASOF, spot=100.0, r=0.04, q=0.0,
                       vol_surface=SkewedVolSurface(atm_level=0.22))


@pytest.fixture
def flat_market():
    return MarketState(asof=ASOF, spot=100.0, r=0.04, q=0.0, vol_surface=FlatVolSurface(0.20))


# --- structure -------------------------------------------------------------

def test_covered_call_is_long_stock_short_call():
    pos = covered_call("XYZ", 100, 105, EXPIRY)
    assert len(pos.legs) == 2
    assert pos.net_shares() == 100
    opt = pos.option_legs[0]
    assert opt.quantity == -1
    assert opt.instrument.is_call


def test_protective_put_is_long_stock_long_put():
    pos = protective_put("XYZ", 100, 95, EXPIRY)
    assert pos.option_legs[0].quantity == 1
    assert not pos.option_legs[0].instrument.is_call


def test_vertical_spread_directions():
    bull = vertical_spread("XYZ", Right.CALL, 100, 110, EXPIRY)
    assert bull.legs[0].quantity > 0 and bull.legs[0].instrument.strike == 100
    assert bull.legs[1].quantity < 0 and bull.legs[1].instrument.strike == 110


def test_iron_condor_enforces_strike_ordering():
    with pytest.raises(ValueError, match="long_put < short_put < short_call < long_call"):
        iron_condor("XYZ", 95, 90, 110, 115, EXPIRY)


def test_strangle_and_collar_enforce_ordering():
    with pytest.raises(ValueError, match="put_strike < call_strike"):
        strangle("XYZ", 110, 90, EXPIRY)
    with pytest.raises(ValueError, match="put_strike < call_strike"):
        collar("XYZ", 100, 110, 90, EXPIRY)


def test_butterfly_structure_is_one_two_one():
    fly = butterfly("XYZ", Right.CALL, 95, 100, 105, EXPIRY)
    quantities = [leg.quantity for leg in fly.legs]
    assert quantities == [1, -2, 1]
    assert sum(quantities) == 0


def test_calendar_spread_requires_ordered_expiries():
    with pytest.raises(ValueError, match="near_expiry < far_expiry"):
        calendar_spread("XYZ", Right.CALL, 100, date(2024, 6, 1), date(2024, 3, 1))


# --- economics -------------------------------------------------------------

def test_short_strangle_collects_a_credit(flat_market):
    pos = strangle("XYZ", 90, 110, EXPIRY, quantity=-1, style=Style.EUROPEAN)
    assert position_value(pos, flat_market) < 0  # negative value == credit received


def test_debit_spread_costs_money(flat_market):
    bull = vertical_spread("XYZ", Right.CALL, 100, 110, EXPIRY, style=Style.EUROPEAN)
    assert position_value(bull, flat_market) > 0


def test_long_butterfly_peaks_at_the_body(flat_market):
    fly = butterfly("XYZ", Right.CALL, 95, 100, 105, EXPIRY, style=Style.EUROPEAN)
    spots = np.array([80.0, 95.0, 100.0, 105.0, 120.0])
    payoff = fly.payoff_at_expiry(spots)
    assert payoff[2] == pytest.approx(500.0)      # peak at the body
    assert payoff[0] == pytest.approx(0.0)
    assert payoff[4] == pytest.approx(0.0)


def test_collar_caps_upside_and_floors_downside():
    pos = collar("XYZ", 100, 95, 110, EXPIRY)
    payoff = pos.payoff_at_expiry(np.array([50.0, 95.0, 110.0, 200.0]))
    assert payoff[0] == pytest.approx(9500.0)   # floored by the put
    assert payoff[3] == pytest.approx(11000.0)  # capped by the call


def test_long_straddle_is_long_gamma_and_vega(flat_market):
    g = position_greeks(straddle("XYZ", 100, EXPIRY, quantity=1, style=Style.EUROPEAN), flat_market)
    assert g.gamma > 0
    assert g.vega > 0
    assert g.theta < 0


# --- delta targeting -------------------------------------------------------

@pytest.mark.parametrize("target", [0.10, 0.20, 0.30, 0.45])
@pytest.mark.parametrize("is_call", [True, False])
def test_strike_for_delta_hits_its_target(market, target, is_call):
    """
    The returned strike must actually have the requested delta, using the same
    surface vol the solver used. Tolerance reflects strike rounding to $1.
    """
    strike = strike_for_delta(market, target, EXPIRY, is_call=is_call, increment=1.0)
    T = (EXPIRY - ASOF).days / 365.0
    from mcop.strategies import option
    contract = option("XYZ", Right.CALL if is_call else Right.PUT, strike, EXPIRY, Style.EUROPEAN)
    sigma = market.iv_for(contract)
    delta = bs_greeks(market.spot, strike, market.r, market.q, sigma, T, is_call).delta
    assert abs(delta) == pytest.approx(target, abs=0.03)


def test_strike_for_delta_is_monotone_in_target(market):
    """A higher call delta must correspond to a lower (more ITM) strike."""
    strikes = [strike_for_delta(market, d, EXPIRY, is_call=True) for d in (0.10, 0.25, 0.40)]
    assert strikes[0] > strikes[1] > strikes[2]


def test_strike_for_delta_rejects_invalid_targets(market):
    for bad in (0.0, 1.0, 1.5, -1.5):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            strike_for_delta(market, bad, EXPIRY, is_call=True)


def test_strike_for_delta_accepts_a_signed_delta_as_a_magnitude(market):
    """
    Puts are quoted with negative delta, so passing -0.20 to mean "the 20-delta
    put" is the natural thing for a caller to do and is treated as a magnitude.
    """
    signed = strike_for_delta(market, -0.20, EXPIRY, is_call=False)
    unsigned = strike_for_delta(market, 0.20, EXPIRY, is_call=False)
    assert signed == unsigned


def test_strike_for_delta_rejects_past_expiry(market):
    with pytest.raises(ValueError, match="expiry must be in the future"):
        strike_for_delta(market, 0.25, date(2023, 1, 1), is_call=True)


def test_round_to_increment():
    assert round_to_increment(102.4, 5.0) == 100.0
    assert round_to_increment(103.0, 5.0) == 105.0
    assert round_to_increment(102.4, 0.0) == 102.4


def test_iron_condor_by_delta_has_correct_ordering_and_wings(market):
    pos = iron_condor_by_delta(market, "XYZ", EXPIRY, short_delta=0.20, wing_width=5.0)
    strikes = [leg.instrument.strike for leg in pos.legs]
    long_put, short_put, short_call, long_call = strikes
    assert long_put < short_put < short_call < long_call
    assert short_put - long_put == pytest.approx(5.0)
    assert long_call - short_call == pytest.approx(5.0)


def test_short_strangle_by_delta_is_short_both_legs(market):
    pos = short_strangle_by_delta(market, "XYZ", EXPIRY, target_delta=0.20)
    assert all(leg.quantity < 0 for leg in pos.legs)
    assert position_greeks(pos, market).theta > 0  # collecting decay


def test_skew_makes_downside_puts_richer(market):
    """Equity skew: a 10% OTM put must imply a higher vol than a 10% OTM call."""
    put_iv = market.vol_surface.iv(90.0, 100.0, 0.25)
    call_iv = market.vol_surface.iv(110.0, 100.0, 0.25)
    assert put_iv > call_iv
