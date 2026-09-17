"""
Exotic payoffs, validated against closed forms and exact identities.

Exotics are harder to test than vanillas because most have no closed form — so
this suite leans on three kinds of check:

1. **Closed forms where they exist** — digitals and geometric Asians.
2. **Exact identities that hold regardless of model** — in-out parity, and the
   decomposition of a vanilla into digitals. These need no second implementation
   and cannot drift.
3. **Convergence behaviour** — the Brownian-bridge barrier correction must
   approach the continuously-monitored price faster than naive sampling does.
"""

import math

import numpy as np
import pytest

from mcop.black_scholes import bs_price
from mcop.exotic_analytic import (
    digital_asset_or_nothing,
    digital_cash_or_nothing,
    down_and_in_call,
    down_and_out_call,
    geometric_asian_moments,
    geometric_asian_price,
)
from mcop.exotics import (
    asian_payoff,
    asian_price_with_control,
    barrier_payoff,
    barrier_survival_probability,
    digital_payoff,
    lookback_payoff,
)
from mcop.greeks import mc_greeks
from mcop.simulate_paths import simulate_gbm_paths

S, K, R, Q, SIGMA, T = 100.0, 100.0, 0.05, 0.0, 0.20, 1.0
N_PATHS, N_STEPS = 200_000, 50
DISC = math.exp(-R * T)


@pytest.fixture(scope="module")
def paths():
    return simulate_gbm_paths(S, R, SIGMA, T, N_STEPS, N_PATHS, q=Q, seed=7, antithetic=True)


def price(payoffs):
    return float(DISC * np.asarray(payoffs, dtype=float).mean())


# --------------------------------------------------------------------------
# Digitals
# --------------------------------------------------------------------------

@pytest.mark.parametrize("is_call", [True, False])
def test_cash_digital_matches_closed_form(paths, is_call):
    mc = price(digital_payoff(paths, K, is_call, cash=1.0))
    cf = digital_cash_or_nothing(S, K, R, Q, SIGMA, T, is_call)
    assert mc == pytest.approx(cf, abs=0.004)


@pytest.mark.parametrize("is_call", [True, False])
def test_asset_digital_matches_closed_form(paths, is_call):
    mc = price(digital_payoff(paths, K, is_call, style="asset"))
    cf = digital_asset_or_nothing(S, K, R, Q, SIGMA, T, is_call)
    assert mc == pytest.approx(cf, abs=0.4)


def test_vanilla_decomposes_into_digitals():
    """
    call = asset-or-nothing call - K * cash-or-nothing call.

    This is an exact identity — it's literally what the two terms of the
    Black-Scholes formula are — so it must hold to machine precision.
    """
    for strike in (80.0, 100.0, 125.0):
        lhs = bs_price(S, strike, R, Q, SIGMA, T, True)
        rhs = (
            digital_asset_or_nothing(S, strike, R, Q, SIGMA, T, True)
            - strike * digital_cash_or_nothing(S, strike, R, Q, SIGMA, T, True)
        )
        assert lhs == pytest.approx(rhs, abs=1e-10)


def test_cash_digital_is_bounded_by_discounted_cash():
    """It pays at most `cash`, so it can never be worth more than its PV."""
    for strike in (60.0, 100.0, 160.0):
        v = digital_cash_or_nothing(S, strike, R, Q, SIGMA, T, True)
        assert 0.0 <= v <= math.exp(-R * T) + 1e-12


def test_digital_price_is_decreasing_in_strike():
    """A higher strike is less likely to be exceeded, so the call is worth less."""
    vals = [digital_cash_or_nothing(S, k, R, Q, SIGMA, T, True) for k in (80, 90, 100, 110, 120)]
    assert all(b < a for a, b in zip(vals, vals[1:]))


def test_expired_digital_is_a_step_function():
    assert digital_cash_or_nothing(110.0, K, R, Q, SIGMA, 0.0, True) == pytest.approx(1.0)
    assert digital_cash_or_nothing(90.0, K, R, Q, SIGMA, 0.0, True) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Asian
# --------------------------------------------------------------------------

def test_geometric_asian_matches_closed_form(paths):
    """
    The closed form uses the same discrete fixing grid as the simulation, so
    agreement should be within Monte Carlo error rather than merely 'close'.
    """
    mc = price(asian_payoff(paths, K, True, "geometric"))
    cf = geometric_asian_price(S, K, R, Q, SIGMA, T, N_STEPS, True)
    assert mc == pytest.approx(cf, abs=0.02)


def test_geometric_asian_moments_tend_to_continuous_limits():
    """As fixings increase, mean time -> T/2 and variance -> sigma^2 T/3."""
    _, var = geometric_asian_moments(S, R, Q, SIGMA, T, n_fixings=20_000)
    assert var == pytest.approx(SIGMA**2 * T / 3.0, rel=1e-3)


def test_asian_is_cheaper_than_vanilla(paths):
    """Averaging suppresses volatility, so an Asian must be worth less."""
    asian = price(asian_payoff(paths, K, True, "arithmetic"))
    vanilla = bs_price(S, K, R, Q, SIGMA, T, True)
    assert asian < vanilla


def test_arithmetic_asian_exceeds_geometric(paths):
    """AM-GM: the arithmetic mean dominates the geometric mean path by path."""
    arith = asian_payoff(paths, K, True, "arithmetic")
    geo = asian_payoff(paths, K, True, "geometric")
    assert np.all(arith >= geo - 1e-12)


def test_control_variate_cuts_the_standard_error(paths):
    """
    The geometric and arithmetic payoffs correlate above 0.99, which is what
    makes the control variate worth using. Verify it genuinely reduces spread
    rather than just returning a similar number.
    """
    plain = DISC * asian_payoff(paths, K, True, "arithmetic")
    geo = DISC * asian_payoff(paths, K, True, "geometric")
    geo_true = geometric_asian_price(S, K, R, Q, SIGMA, T, N_STEPS, True)

    corr = float(np.corrcoef(plain, geo)[0, 1])
    assert corr > 0.99

    controlled, beta = asian_price_with_control(paths, K, R, T, SIGMA, True, Q)

    # Residual spread after removing the fitted geometric component.
    residual = plain - beta * (geo - geo_true)
    assert residual.std(ddof=1) < 0.25 * plain.std(ddof=1)

    # And it must stay unbiased: both estimate the same quantity.
    se_plain = plain.std(ddof=1) / math.sqrt(plain.size)
    assert controlled == pytest.approx(plain.mean(), abs=5 * se_plain)


def test_asian_rejects_unknown_averaging(paths):
    with pytest.raises(ValueError, match="arithmetic.*geometric"):
        asian_payoff(paths, K, True, "harmonic")


# --------------------------------------------------------------------------
# Lookback
# --------------------------------------------------------------------------

def test_floating_lookback_is_never_negative(paths):
    """The holder buys at the minimum, so the payoff cannot go against them."""
    assert np.all(lookback_payoff(paths, is_call=True) >= -1e-12)
    assert np.all(lookback_payoff(paths, is_call=False) >= -1e-12)


def test_lookback_costs_more_than_vanilla(paths):
    """Hindsight is worth paying for."""
    lb = price(lookback_payoff(paths, is_call=True))
    assert lb > bs_price(S, K, R, Q, SIGMA, T, True)


def test_fixed_strike_lookback_dominates_vanilla(paths):
    """
    max over the path is at least the terminal value, so the fixed-strike
    lookback payoff dominates the vanilla payoff path by path.
    """
    lb = lookback_payoff(paths, is_call=True, K=K)
    vanilla = np.maximum(paths[:, -1] - K, 0.0)
    assert np.all(lb >= vanilla - 1e-12)


# --------------------------------------------------------------------------
# Barrier
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bridge", [False, True])
@pytest.mark.parametrize("direction,H", [("down", 90.0), ("up", 115.0)])
def test_in_out_parity_is_exact(paths, bridge, direction, H):
    """
    knock-in + knock-out = vanilla, because every path either touched the
    barrier or did not. Holds for both monitoring treatments, and to machine
    precision — this is the strongest barrier test available without a second
    closed form.
    """
    kw = dict(sigma=SIGMA, T=T, bridge=bridge)
    out = barrier_payoff(paths, K, H, True, direction, "out", **kw)
    inn = barrier_payoff(paths, K, H, True, direction, "in", **kw)
    vanilla = np.maximum(paths[:, -1] - K, 0.0)
    np.testing.assert_allclose(out + inn, vanilla, rtol=1e-12, atol=1e-10)


def test_bridge_reduces_discretisation_bias():
    """
    Discrete sampling misses barrier crossings between observation times, which
    prices knock-outs too high. The Brownian-bridge correction should land much
    closer to the continuously-monitored closed form at the same step count.
    """
    H = 90.0
    reference = down_and_out_call(S, K, H, R, Q, SIGMA, T)
    p = simulate_gbm_paths(S, R, SIGMA, T, 25, 100_000, q=Q, seed=11, antithetic=True)

    naive = price(barrier_payoff(p, K, H, True, "down", "out"))
    bridged = price(barrier_payoff(p, K, H, True, "down", "out", sigma=SIGMA, T=T, bridge=True))

    assert naive > reference                      # biased high, as predicted
    assert abs(bridged - reference) < 0.25 * abs(naive - reference)


def test_naive_barrier_bias_shrinks_with_more_steps():
    """The bias is real but converges — slowly — as sampling gets finer."""
    H = 90.0
    reference = down_and_out_call(S, K, H, R, Q, SIGMA, T)
    errs = []
    for steps in (25, 100):
        p = simulate_gbm_paths(S, R, SIGMA, T, steps, 100_000, q=Q, seed=11, antithetic=True)
        errs.append(price(barrier_payoff(p, K, H, True, "down", "out")) - reference)
    assert errs[1] < errs[0]


def test_knock_out_is_cheaper_than_vanilla(paths):
    out = price(barrier_payoff(paths, K, 90.0, True, "down", "out"))
    assert out < bs_price(S, K, R, Q, SIGMA, T, True)


def test_knock_out_value_falls_as_barrier_approaches_spot(paths):
    """A nearer barrier is easier to hit, so the knock-out is worth less."""
    vals = [price(barrier_payoff(paths, K, h, True, "down", "out")) for h in (70.0, 85.0, 95.0)]
    assert vals[0] > vals[1] > vals[2]


def test_survival_probability_is_a_probability(paths):
    surv = barrier_survival_probability(paths, 90.0, "down", SIGMA, T)
    assert np.all(surv >= 0.0) and np.all(surv <= 1.0)


def test_survival_is_zero_once_the_barrier_is_breached(paths):
    """Paths that actually closed below the barrier cannot have survived."""
    surv = barrier_survival_probability(paths, 90.0, "down", SIGMA, T)
    breached = (paths[:, 1:] <= 90.0).any(axis=1)
    assert np.all(surv[breached] == 0.0)


def test_analytic_barrier_respects_in_out_parity():
    """The closed forms must satisfy the same identity as the simulation."""
    H = 90.0
    vanilla = bs_price(S, K, R, Q, SIGMA, T, True)
    total = down_and_in_call(S, K, H, R, Q, SIGMA, T) + down_and_out_call(S, K, H, R, Q, SIGMA, T)
    assert total == pytest.approx(vanilla, abs=1e-10)


def test_analytic_barrier_refuses_unsupported_geometry():
    # Barrier above the strike needs a different expression.
    with pytest.raises(ValueError, match="at or below the strike"):
        down_and_in_call(S, 90.0, 95.0, R, Q, SIGMA, T)
    # Barrier below the strike but at/above spot is not a live down-barrier:
    # it would already be breached at inception.
    with pytest.raises(ValueError, match="below spot"):
        down_and_in_call(S, 120.0, 110.0, R, Q, SIGMA, T)


def test_barrier_rejects_bridge_without_parameters(paths):
    with pytest.raises(ValueError, match="requires sigma and T"):
        barrier_payoff(paths, K, 90.0, True, "down", "out", bridge=True)


# --------------------------------------------------------------------------
# Greeks on exotic payoffs
# --------------------------------------------------------------------------

def _digital_delta_closed_form(S_, K_, r_, q_, sigma_, T_):
    """d/dS of exp(-rT) N(d2) = exp(-rT) * phi(d2) / (S * sigma * sqrt(T))."""
    d2 = (math.log(S_ / K_) + (r_ - q_ - 0.5 * sigma_**2) * T_) / (sigma_ * math.sqrt(T_))
    phi = math.exp(-0.5 * d2 * d2) / math.sqrt(2.0 * math.pi)
    return math.exp(-r_ * T_) * phi / (S_ * sigma_ * math.sqrt(T_))


def test_likelihood_ratio_handles_the_digital_delta():
    """
    The payoff that justifies carrying three Greek estimators.

    A digital's payoff is a step function. The likelihood-ratio estimator never
    differentiates the payoff — it differentiates the density — so it copes.
    """
    ref = _digital_delta_closed_form(S, K, R, Q, SIGMA, T)
    g = mc_greeks(
        S, K, R, SIGMA, T, True, q=Q, n_paths=400_000, n_steps=25, seed=3,
        method="lr", payoff=lambda p: digital_payoff(p, K, True, cash=1.0),
    )
    assert g.delta == pytest.approx(ref, rel=0.10)
    assert g.price == pytest.approx(digital_cash_or_nothing(S, K, R, Q, SIGMA, T, True), abs=0.004)


def test_finite_difference_also_handles_the_digital(paths):
    """FD treats the payoff as a black box, so it copes too — just more noisily."""
    ref = _digital_delta_closed_form(S, K, R, Q, SIGMA, T)
    g = mc_greeks(
        S, K, R, SIGMA, T, True, q=Q, n_paths=400_000, n_steps=25, seed=3,
        method="fd", rel_bump=0.02,
        payoff=lambda p: digital_payoff(p, K, True, cash=1.0),
    )
    assert g.delta == pytest.approx(ref, rel=0.25)


def test_pathwise_refuses_a_custom_payoff():
    """
    It cannot differentiate an opaque callable, and returning a plausible but
    wrong number would be worse than refusing.
    """
    with pytest.raises(ValueError, match="cannot take a custom payoff"):
        mc_greeks(S, K, R, SIGMA, T, True, method="pathwise",
                  payoff=lambda p: digital_payoff(p, K, True))


def test_custom_payoff_refused_for_american_exercise():
    with pytest.raises(ValueError, match="not supported with American exercise"):
        mc_greeks(S, K, R, SIGMA, T, False, style="american", method="fd",
                  payoff=lambda p: digital_payoff(p, K, False))


def test_greeks_on_a_path_dependent_payoff():
    """
    Asian Greeks via finite difference. No closed form to check against, so the
    assertions are structural: an Asian call is long delta and long vega, and
    its delta is lower than the vanilla's because averaging damps the payoff's
    response to spot.
    """
    asian = mc_greeks(
        S, K, R, SIGMA, T, True, q=Q, n_paths=100_000, n_steps=25, seed=5,
        method="fd", payoff=lambda p: asian_payoff(p, K, True, "arithmetic"),
    )
    vanilla = mc_greeks(S, K, R, SIGMA, T, True, q=Q, n_paths=100_000, n_steps=25, seed=5)

    assert 0.0 < asian.delta < vanilla.delta
    assert asian.vega > 0.0
    assert asian.price < vanilla.price


# --------------------------------------------------------------------------
# Conventions
# --------------------------------------------------------------------------

def test_fixings_exclude_the_known_starting_spot():
    """
    n simulated steps means n fixings. Including the known S0 in an average is a
    different contract, and mixing the two silently is a classic pricing bug —
    so pin the convention.
    """
    p = simulate_gbm_paths(S, R, SIGMA, T, 4, 3, q=Q, seed=1)
    expected = p[:, 1:].mean(axis=1)
    got = asian_payoff(p, 0.0, True, "arithmetic")  # zero strike => payoff is the average
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_payoffs_reject_degenerate_paths():
    with pytest.raises(ValueError, match="at least one simulated step"):
        asian_payoff(np.array([[100.0]]), K, True)
