"""The position data model: signs, multipliers, expiry handling, netting."""

from datetime import date

import numpy as np
import pytest

from mcop.instruments import Leg, OptionContract, Position, Right, Style, Underlying

ASOF = date(2024, 1, 15)
EXPIRY = date(2024, 3, 15)


def call(strike=100.0, expiry=EXPIRY):
    return OptionContract("XYZ", Right.CALL, strike, expiry, Style.AMERICAN, 100.0)


def put(strike=100.0, expiry=EXPIRY):
    return OptionContract("XYZ", Right.PUT, strike, expiry, Style.AMERICAN, 100.0)


def test_time_to_expiry_is_calendar_based():
    c = call()
    assert c.days_to_expiry(ASOF) == 60
    assert c.time_to_expiry(ASOF) == pytest.approx(60 / 365.0)


def test_expired_contract_floors_at_zero_not_negative():
    """A negative maturity would produce NaNs in every pricer downstream."""
    c = call(expiry=date(2024, 1, 1))
    assert c.time_to_expiry(ASOF) == 0.0
    assert c.days_to_expiry(ASOF) == 0


def test_intrinsic_values():
    assert call(100).intrinsic(120) == 20.0
    assert call(100).intrinsic(80) == 0.0
    assert put(100).intrinsic(80) == 20.0
    assert put(100).intrinsic(120) == 0.0


def test_signed_shares_applies_multiplier_and_direction():
    assert Leg(call(), 2).signed_shares == 200.0
    assert Leg(call(), -3).signed_shares == -300.0
    assert Leg(Underlying("XYZ"), 100).signed_shares == 100.0


def test_zero_quantity_leg_is_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        Leg(call(), 0)


def test_invalid_contract_parameters_rejected():
    with pytest.raises(ValueError, match="strike"):
        OptionContract("XYZ", Right.CALL, -5.0, EXPIRY)
    with pytest.raises(ValueError, match="multiplier"):
        OptionContract("XYZ", Right.CALL, 100.0, EXPIRY, multiplier=0.0)


def test_short_call_payoff_is_negative_above_strike():
    leg = Leg(call(100), -1)
    assert leg.payoff_at_expiry(120.0) == pytest.approx(-2000.0)
    assert leg.payoff_at_expiry(90.0) == pytest.approx(0.0)


def test_covered_call_payoff_caps_above_strike():
    """Long 100 shares plus a short 105 call cannot be worth more than 10,500."""
    pos = Position(
        name="covered call",
        legs=[Leg(Underlying("XYZ"), 100), Leg(call(105), -1)],
    )
    spots = np.array([90.0, 105.0, 130.0, 200.0])
    payoff = pos.payoff_at_expiry(spots)
    assert payoff[0] == pytest.approx(9000.0)
    assert payoff[1] == pytest.approx(10500.0)
    assert payoff[2] == pytest.approx(10500.0)
    assert payoff[3] == pytest.approx(10500.0)


def test_multi_expiry_payoff_diagram_is_refused():
    """A calendar has no single expiry, so the diagram is undefined."""
    pos = Position(
        name="calendar",
        legs=[Leg(call(100, EXPIRY), -1), Leg(call(100, date(2024, 6, 21)), 1)],
    )
    assert not pos.is_single_expiry
    with pytest.raises(ValueError, match="single expiry payoff diagram is not defined"):
        pos.payoff_at_expiry(np.array([100.0]))


def test_empty_position_rejected():
    with pytest.raises(ValueError, match="at least one leg"):
        Position(legs=[])


def test_position_partitions_legs():
    pos = Position(legs=[Leg(Underlying("XYZ"), 100), Leg(call(105), -1), Leg(put(95), -1)])
    assert len(pos.option_legs) == 2
    assert len(pos.underlying_legs) == 1
    assert pos.net_shares() == 100.0
    assert pos.is_single_expiry
