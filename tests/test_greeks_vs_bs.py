"""
Monte Carlo Greeks validated against the closed form.

Tolerances are set from the estimator's own noise level rather than picked to
pass: delta from 200k antithetic paths is good to a few units in the third
decimal, gamma is an order of magnitude noisier because it is a second
difference, and the likelihood-ratio estimator is noisier still by construction.
"""

import math

import numpy as np
import pytest

from mcop.black_scholes import bs_greeks
from mcop.greeks import mc_greeks
from mcop.simulate_paths import draw_normals, gbm_paths_from_normals, simulate_gbm_paths

S, K, R, Q, SIGMA, T = 100.0, 100.0, 0.05, 0.0, 0.20, 1.0
COMMON = dict(n_paths=200_000, n_steps=50, seed=11, q=Q)


@pytest.mark.parametrize("is_call", [True, False])
def test_fd_greeks_match_black_scholes(is_call):
    ref = bs_greeks(S, K, R, Q, SIGMA, T, is_call)
    got = mc_greeks(S, K, R, SIGMA, T, is_call, method="fd", **COMMON)

    assert got.price == pytest.approx(ref.price, abs=0.05)
    assert got.delta == pytest.approx(ref.delta, abs=0.005)
    assert got.gamma == pytest.approx(ref.gamma, abs=0.002)
    assert got.vega == pytest.approx(ref.vega, abs=0.5)
    assert got.theta == pytest.approx(ref.theta, abs=0.1)
    assert got.rho == pytest.approx(ref.rho, abs=0.5)


@pytest.mark.parametrize("is_call", [True, False])
def test_pathwise_greeks_match_black_scholes(is_call):
    ref = bs_greeks(S, K, R, Q, SIGMA, T, is_call)
    got = mc_greeks(S, K, R, SIGMA, T, is_call, method="pathwise", **COMMON)

    assert got.delta == pytest.approx(ref.delta, abs=0.005)
    assert got.vega == pytest.approx(ref.vega, abs=0.5)
    # Pathwise cannot produce gamma for a vanilla payoff and says so explicitly.
    assert math.isnan(got.gamma)


@pytest.mark.parametrize("is_call", [True, False])
def test_likelihood_ratio_greeks_match_black_scholes(is_call):
    ref = bs_greeks(S, K, R, Q, SIGMA, T, is_call)
    got = mc_greeks(S, K, R, SIGMA, T, is_call, method="lr", **COMMON)

    # LR is the noisiest estimator, so tolerances are correspondingly wider.
    assert got.delta == pytest.approx(ref.delta, abs=0.01)
    assert got.gamma == pytest.approx(ref.gamma, abs=0.004)
    assert got.vega == pytest.approx(ref.vega, abs=1.5)


def test_common_random_numbers_beat_independent_draws():
    """
    The central claim of the FD implementation: reusing shocks across bumps
    collapses the variance of the difference.

    Measured by repeating the delta estimate with different seeds and comparing
    the spread of the results. With independent draws per bump the estimator is
    dramatically noisier at the same path count.
    """
    h = 1.0

    def delta_crn(seed):
        Z = draw_normals(50, 20_000, seed=seed, antithetic=True)
        up = gbm_paths_from_normals(Z, S + h, R, SIGMA, T, Q)[:, -1]
        dn = gbm_paths_from_normals(Z, S - h, R, SIGMA, T, Q)[:, -1]
        disc = math.exp(-R * T)
        return disc * (np.maximum(up - K, 0).mean() - np.maximum(dn - K, 0).mean()) / (2 * h)

    def delta_independent(seed):
        # Different shocks for each bump: the naive implementation.
        Zu = draw_normals(50, 20_000, seed=seed, antithetic=True)
        Zd = draw_normals(50, 20_000, seed=seed + 10_000, antithetic=True)
        up = gbm_paths_from_normals(Zu, S + h, R, SIGMA, T, Q)[:, -1]
        dn = gbm_paths_from_normals(Zd, S - h, R, SIGMA, T, Q)[:, -1]
        disc = math.exp(-R * T)
        return disc * (np.maximum(up - K, 0).mean() - np.maximum(dn - K, 0).mean()) / (2 * h)

    seeds = range(12)
    sd_crn = float(np.std([delta_crn(s) for s in seeds], ddof=1))
    sd_ind = float(np.std([delta_independent(s) for s in seeds], ddof=1))

    assert sd_crn < sd_ind / 10.0, f"CRN sd={sd_crn:.5f} vs independent sd={sd_ind:.5f}"


def test_american_put_delta_is_more_negative_than_european():
    """
    Sanity check on the American branch: the right to exercise early makes an
    in-the-money put more sensitive to spot than its European twin.
    """
    euro = mc_greeks(95.0, K, R, SIGMA, T, False, style="european", method="fd",
                     n_paths=60_000, n_steps=50, seed=5)
    amer = mc_greeks(95.0, K, R, SIGMA, T, False, style="american", method="fd",
                     n_paths=60_000, n_steps=50, seed=5)
    assert amer.delta < euro.delta
    assert amer.price > euro.price


def test_american_rejects_non_fd_methods():
    with pytest.raises(ValueError, match="only supports method='fd'"):
        mc_greeks(S, K, R, SIGMA, T, False, style="american", method="pathwise")


def test_refactored_simulator_is_unchanged():
    """
    draw_normals + gbm_paths_from_normals must reproduce simulate_gbm_paths
    exactly, so splitting the simulator did not perturb any existing result.
    """
    direct = simulate_gbm_paths(S, R, SIGMA, T, 25, 1000, q=Q, seed=99, antithetic=True)
    Z = draw_normals(25, 1000, seed=99, antithetic=True)
    composed = gbm_paths_from_normals(Z, S, R, SIGMA, T, Q)
    assert np.array_equal(direct, composed)
