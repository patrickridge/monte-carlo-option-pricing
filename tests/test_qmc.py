"""
Quasi-Monte Carlo: correctness of the construction, and that it actually helps.

Two very different kinds of test here. The first group checks that swapping
pseudo-random draws for a Sobol sequence has not changed *what is being
computed* — the distribution of the paths must be identical, or the prices are
simply wrong. The second group checks the thing QMC exists for: faster
convergence. That one is asserted numerically rather than assumed, because a
QMC implementation that is subtly wrong usually still runs, still produces
plausible prices, and simply fails to converge any faster than what it replaced.
"""

import math

import numpy as np
import pytest

from mcop.black_scholes import bs_price
from mcop.exotics import asian_payoff
from mcop.qmc import bridge_order, qmc_price, rqmc_estimate, sobol_normals
from mcop.qmc import _bridge_from_normals
from mcop.simulate_paths import draw_normals, gbm_paths_from_normals

S, K, R, Q, SIGMA, T = 100.0, 100.0, 0.05, 0.0, 0.20, 1.0
DISC = math.exp(-R * T)


def _european_call_price(Z):
    ST = gbm_paths_from_normals(Z, S0=S, r=R, sigma=SIGMA, T=T, q=Q)[:, -1]
    return float(DISC * np.maximum(ST - K, 0.0).mean())


# --------------------------------------------------------------------------
# The bridge construction
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_steps", [2, 4, 8, 16, 50, 100, 252])
def test_bridge_schedule_fills_every_time_index_exactly_once(n_steps):
    idx, _, _, _ = bridge_order(n_steps)
    assert sorted(idx.tolist()) == list(range(n_steps))


@pytest.mark.parametrize("n_steps", [4, 16, 100])
def test_bridge_uses_its_best_dimension_on_the_terminal_point(n_steps):
    """
    The whole point of the construction: the first Sobol dimension is the best
    distributed, so it must drive the feature carrying the most variance — the
    endpoint of the path.
    """
    idx, left, right, sd = bridge_order(n_steps)
    assert idx[0] == n_steps - 1
    assert right[0] == -1                      # conditioned only on W(0) = 0
    assert sd[0] == pytest.approx(math.sqrt(n_steps))


def test_bridge_preserves_the_distribution():
    """
    The construction must be a pure re-ordering of which input drives which
    feature. If it changes the law of the process, every price computed through
    it is wrong.

    Checked against the defining property of Brownian motion:
    Cov(W_i, W_j) = min(i, j).
    """
    n_steps, n_paths = 8, 300_000
    rng = np.random.default_rng(0)
    Z = _bridge_from_normals(rng.standard_normal((n_paths, n_steps)))

    W = np.cumsum(Z, axis=1)
    cov = np.cov(W.T)
    theory = np.minimum.outer(np.arange(1, n_steps + 1), np.arange(1, n_steps + 1))

    assert np.abs(cov - theory).max() < 0.06
    assert Z.mean() == pytest.approx(0.0, abs=0.01)
    assert Z.var() == pytest.approx(1.0, abs=0.01)


def test_bridge_increments_are_standardised():
    """Output must be unit-variance per step, so it drops into the GBM builder."""
    rng = np.random.default_rng(1)
    Z = _bridge_from_normals(rng.standard_normal((200_000, 16)))
    per_step_var = Z.var(axis=0)
    np.testing.assert_allclose(per_step_var, np.ones(16), atol=0.02)


# --------------------------------------------------------------------------
# Sobol normals
# --------------------------------------------------------------------------

def test_sobol_normals_shape_and_moments():
    Z = sobol_normals(50, 4096, seed=7)
    assert Z.shape == (4096, 50)
    assert Z.mean() == pytest.approx(0.0, abs=1e-3)
    assert Z.var() == pytest.approx(1.0, abs=1e-2)


def test_sobol_rounds_up_to_a_power_of_two_then_trims():
    """
    Sobol sequences are balanced in blocks of 2^m. Asking for 3000 points and
    taking the first 3000 of 4096 keeps the requested count without silently
    using an unbalanced prefix of a differently-sized sequence.
    """
    assert sobol_normals(10, 3000, seed=1).shape == (3000, 10)
    assert sobol_normals(10, 4096, seed=1).shape == (4096, 10)


def test_scrambling_changes_the_sequence_but_not_its_quality():
    a = sobol_normals(20, 2048, seed=1, scramble=True)
    b = sobol_normals(20, 2048, seed=2, scramble=True)
    assert not np.allclose(a, b)
    for Z in (a, b):
        assert Z.mean() == pytest.approx(0.0, abs=5e-3)
        assert Z.var() == pytest.approx(1.0, abs=2e-2)


def test_unscrambled_sobol_is_deterministic():
    a = sobol_normals(8, 1024, seed=1, scramble=False)
    b = sobol_normals(8, 1024, seed=99, scramble=False)
    np.testing.assert_allclose(a, b)


def test_sobol_prices_agree_with_the_closed_form():
    """Different sampling scheme, same answer — this is the correctness check."""
    exact = bs_price(S, K, R, Q, SIGMA, T, True)
    got = _european_call_price(sobol_normals(50, 16384, seed=3))
    assert got == pytest.approx(exact, abs=0.01)


def test_sobol_rejects_degenerate_sizes():
    with pytest.raises(ValueError, match="must be positive"):
        sobol_normals(0, 128)
    with pytest.raises(ValueError, match="must be positive"):
        sobol_normals(8, 0)


# --------------------------------------------------------------------------
# The thing QMC exists for
# --------------------------------------------------------------------------

def test_qmc_converges_faster_than_pseudo_random():
    """
    The headline claim, asserted rather than assumed.

    A broken QMC implementation typically still produces plausible prices and
    simply fails to beat pseudo-random. Averaging |error| over several seeds at
    a fixed path count is the direct test.
    """
    exact = bs_price(S, K, R, Q, SIGMA, T, True)
    n = 8192

    pseudo = np.mean([abs(_european_call_price(draw_normals(50, n, seed=s)) - exact)
                      for s in range(6)])
    sobol = np.mean([abs(_european_call_price(sobol_normals(50, n, seed=s)) - exact)
                     for s in range(6)])

    assert sobol < pseudo / 5.0, f"sobol err {sobol:.5f} vs pseudo {pseudo:.5f}"


def test_brownian_bridge_matters_in_high_dimensions():
    """
    Sobol points are far better distributed in early dimensions than late ones.
    With 50 time steps the bridge construction — which puts the good dimensions
    on the large-scale features — should clearly beat the naive step-by-step
    mapping. In low dimensions the difference should largely disappear.
    """
    exact = bs_price(S, K, R, Q, SIGMA, T, True)
    n = 8192

    plain = np.mean([abs(_european_call_price(sobol_normals(50, n, seed=s, bridge=False)) - exact)
                     for s in range(6)])
    bridged = np.mean([abs(_european_call_price(sobol_normals(50, n, seed=s, bridge=True)) - exact)
                       for s in range(6)])

    assert bridged < plain / 3.0, f"bridged {bridged:.5f} vs plain {plain:.5f}"


def test_qmc_error_shrinks_faster_than_one_over_sqrt_n():
    """
    Pseudo-random error falls like 1/sqrt(N): 16x the paths buys 4x accuracy.
    QMC should do materially better than that factor of four.
    """
    exact = bs_price(S, K, R, Q, SIGMA, T, True)
    small = np.mean([abs(_european_call_price(sobol_normals(50, 2048, seed=s)) - exact)
                     for s in range(6)])
    large = np.mean([abs(_european_call_price(sobol_normals(50, 32768, seed=s)) - exact)
                     for s in range(6)])
    assert small / large > 6.0, f"improvement was only {small / large:.1f}x"


# --------------------------------------------------------------------------
# Randomised QMC and error estimation
# --------------------------------------------------------------------------

def test_rqmc_reports_an_error_bar_that_brackets_the_truth():
    """
    QMC on its own gives a number with no confidence interval, because the
    randomness a standard error is computed from has been removed. Randomised
    QMC restores it by averaging over independent scrambles.
    """
    exact = bs_price(S, K, R, Q, SIGMA, T, True)
    est, se = rqmc_estimate(_european_call_price, 50, 4096, n_scrambles=16, seed=0)

    assert se > 0.0
    assert abs(est - exact) < 4.0 * se + 1e-4
    assert abs(est - exact) < 0.01


def test_rqmc_error_bar_is_tighter_than_plain_monte_carlo():
    """The error estimate should be small, not merely present."""
    _, se_q = rqmc_estimate(_european_call_price, 50, 4096, n_scrambles=16, seed=0)

    vals = np.array([_european_call_price(draw_normals(50, 4096, seed=s)) for s in range(16)])
    se_p = vals.std(ddof=1) / math.sqrt(16)

    assert se_q < se_p / 3.0, f"rqmc se {se_q:.5f} vs pseudo se {se_p:.5f}"


def test_rqmc_needs_at_least_two_scrambles():
    with pytest.raises(ValueError, match="at least 2 scrambles"):
        rqmc_estimate(_european_call_price, 10, 256, n_scrambles=1)


def test_qmc_price_on_a_path_dependent_payoff():
    """
    Asian options are where QMC should shine: smooth, path-dependent, and
    high-dimensional. Checked against a high-path pseudo-random reference.
    """
    payoff = lambda p: asian_payoff(p, K, True, "arithmetic")  # noqa: E731
    est, se = qmc_price(payoff, S, R, SIGMA, T, n_steps=50, n_paths=4096,
                        q=Q, n_scrambles=12, seed=1)

    ref_paths = gbm_paths_from_normals(draw_normals(50, 400_000, seed=9, antithetic=True),
                                       S0=S, r=R, sigma=SIGMA, T=T, q=Q)
    reference = float(DISC * asian_payoff(ref_paths, K, True, "arithmetic").mean())

    assert se > 0.0
    assert est == pytest.approx(reference, abs=0.05)
