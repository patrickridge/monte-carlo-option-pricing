"""Closed-form pricer: parity, bounds, degenerate inputs, and IV inversion."""

import math

import pytest

from mcop.black_scholes import bs_price, bs_greeks, implied_vol

S, K, R, Q, SIGMA, T = 100.0, 100.0, 0.05, 0.02, 0.20, 1.0


def test_put_call_parity():
    """C - P = S*exp(-qT) - K*exp(-rT) must hold to machine precision."""
    c = bs_price(S, K, R, Q, SIGMA, T, True)
    p = bs_price(S, K, R, Q, SIGMA, T, False)
    expected = S * math.exp(-Q * T) - K * math.exp(-R * T)
    assert c - p == pytest.approx(expected, abs=1e-10)


def test_prices_respect_no_arbitrage_bounds():
    c = bs_price(S, K, R, Q, SIGMA, T, True)
    intrinsic_fwd = max(S * math.exp(-Q * T) - K * math.exp(-R * T), 0.0)
    assert intrinsic_fwd <= c <= S * math.exp(-Q * T)


def test_price_is_increasing_in_vol():
    """Monotonicity in sigma is what makes the bisection IV solver valid."""
    prices = [bs_price(S, K, R, Q, v, T, True) for v in (0.05, 0.15, 0.30, 0.60)]
    assert all(b > a for a, b in zip(prices, prices[1:]))


def test_expired_option_is_worth_intrinsic():
    assert bs_price(120.0, K, R, Q, SIGMA, 0.0, True) == pytest.approx(20.0)
    assert bs_price(80.0, K, R, Q, SIGMA, 0.0, True) == pytest.approx(0.0)
    assert bs_price(80.0, K, R, Q, SIGMA, 0.0, False) == pytest.approx(20.0)


def test_zero_vol_is_discounted_forward_intrinsic():
    fwd = S * math.exp((R - Q) * T)
    expected = math.exp(-R * T) * max(fwd - K, 0.0)
    assert bs_price(S, K, R, Q, 0.0, T, True) == pytest.approx(expected)


def test_gamma_and_vega_match_across_call_and_put():
    """Parity is linear in S, so it vanishes under d2/dS2 and d/dsigma."""
    gc = bs_greeks(S, K, R, Q, SIGMA, T, True)
    gp = bs_greeks(S, K, R, Q, SIGMA, T, False)
    assert gc.gamma == pytest.approx(gp.gamma, rel=1e-12)
    assert gc.vega == pytest.approx(gp.vega, rel=1e-12)


def test_delta_parity():
    """Call delta - put delta = exp(-qT)."""
    gc = bs_greeks(S, K, R, Q, SIGMA, T, True)
    gp = bs_greeks(S, K, R, Q, SIGMA, T, False)
    assert gc.delta - gp.delta == pytest.approx(math.exp(-Q * T), abs=1e-10)


def test_greeks_match_numerical_derivatives():
    """The analytic formulas must equal a finite difference of the pricer."""
    g = bs_greeks(S, K, R, Q, SIGMA, T, True)
    h = 1e-4

    fd_delta = (bs_price(S + h, K, R, Q, SIGMA, T, True) - bs_price(S - h, K, R, Q, SIGMA, T, True)) / (2 * h)
    fd_gamma = (
        bs_price(S + h, K, R, Q, SIGMA, T, True)
        - 2 * bs_price(S, K, R, Q, SIGMA, T, True)
        + bs_price(S - h, K, R, Q, SIGMA, T, True)
    ) / h**2
    fd_vega = (bs_price(S, K, R, Q, SIGMA + h, T, True) - bs_price(S, K, R, Q, SIGMA - h, T, True)) / (2 * h)
    fd_rho = (bs_price(S, K, R + h, Q, SIGMA, T, True) - bs_price(S, K, R - h, Q, SIGMA, T, True)) / (2 * h)
    # theta is dV/dt = -dV/dT
    fd_theta = -(bs_price(S, K, R, Q, SIGMA, T + h, True) - bs_price(S, K, R, Q, SIGMA, T - h, True)) / (2 * h)

    assert g.delta == pytest.approx(fd_delta, abs=1e-6)
    assert g.gamma == pytest.approx(fd_gamma, abs=1e-4)
    assert g.vega == pytest.approx(fd_vega, abs=1e-4)
    assert g.rho == pytest.approx(fd_rho, abs=1e-4)
    assert g.theta == pytest.approx(fd_theta, abs=1e-4)


def test_long_call_theta_is_negative():
    """Sign convention: value decays as calendar time passes."""
    assert bs_greeks(S, K, R, 0.0, SIGMA, T, True).theta < 0


def test_greeks_scaling_conventions():
    g = bs_greeks(S, K, R, Q, SIGMA, T, True)
    s = g.scaled()
    assert s.vega == pytest.approx(g.vega / 100.0)
    assert s.theta == pytest.approx(g.theta / 365.0)


def test_greeks_are_additive_and_scalable():
    g = bs_greeks(S, K, R, Q, SIGMA, T, True)
    assert (g + g).delta == pytest.approx(2 * g.delta)
    assert (3 * g).vega == pytest.approx(3 * g.vega)


@pytest.mark.parametrize("target_vol", [0.05, 0.12, 0.20, 0.45, 0.90])
@pytest.mark.parametrize("strike", [80.0, 100.0, 130.0])
def test_implied_vol_round_trip(target_vol, strike):
    price = bs_price(S, strike, R, Q, target_vol, T, True)
    assert implied_vol(price, S, strike, R, Q, T, True) == pytest.approx(target_vol, abs=1e-5)


def test_implied_vol_returns_nan_outside_arbitrage_bounds():
    """An unexplainable price is a data problem the caller must see."""
    assert math.isnan(implied_vol(1e6, S, K, R, Q, T, True))
    assert math.isnan(implied_vol(-1.0, S, K, R, Q, T, True))


def test_implied_vol_on_expired_option_is_nan():
    assert math.isnan(implied_vol(5.0, S, K, R, Q, 0.0, True))
