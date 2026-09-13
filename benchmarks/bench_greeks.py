"""
Benchmark: Monte Carlo Greek estimators.

Two questions this answers with measurements rather than assertion:

1. How much does using common random numbers actually improve a finite
   difference Greek? (Answer: roughly two orders of magnitude in standard
   deviation at a 1% bump.)
2. How do the three estimators compare on accuracy, variance, and cost?

Run:  python benchmarks/bench_greeks.py
"""

from __future__ import annotations

import math
import time

import numpy as np

from mcop.black_scholes import bs_greeks
from mcop.greeks import mc_greeks
from mcop.simulate_paths import draw_normals, gbm_paths_from_normals

S0, K, R, Q, SIGMA, T = 100.0, 100.0, 0.05, 0.0, 0.20, 1.0
N_PATHS, N_STEPS, N_TRIALS = 50_000, 50, 25


def _european_call_price(Z, S0_, sigma_=SIGMA, T_=T):
    ST = gbm_paths_from_normals(Z, S0_, R, sigma_, T_, Q)[:, -1]
    return math.exp(-R * T_) * float(np.maximum(ST - K, 0.0).mean())


def crn_vs_independent(bumps=(0.01, 0.005, 0.001)) -> None:
    """
    Estimate delta many times and compare the spread of the estimates.

    Shared shocks make the two bumped prices move together, so their difference
    is almost noiseless. Independent shocks leave two uncorrelated errors of
    size ~sigma_MC each, and dividing by a small 2h amplifies them — the smaller
    the bump, the worse it gets, which is the trap.
    """
    ref = bs_greeks(S0, K, R, Q, SIGMA, T, True).delta
    print(f"\nreference delta (Black-Scholes): {ref:.6f}")
    print(f"{'bump':>8} {'method':>13} {'mean':>10} {'std dev':>11} {'bias':>10}")
    print("-" * 56)

    for rel in bumps:
        h = rel * S0
        crn, ind = [], []
        for trial in range(N_TRIALS):
            Z = draw_normals(N_STEPS, N_PATHS, seed=trial, antithetic=True)
            crn.append((_european_call_price(Z, S0 + h) - _european_call_price(Z, S0 - h)) / (2 * h))

            Z_up = draw_normals(N_STEPS, N_PATHS, seed=trial, antithetic=True)
            Z_dn = draw_normals(N_STEPS, N_PATHS, seed=trial + 100_000, antithetic=True)
            ind.append((_european_call_price(Z_up, S0 + h) - _european_call_price(Z_dn, S0 - h)) / (2 * h))

        for label, vals in (("common RN", crn), ("independent", ind)):
            arr = np.array(vals)
            print(f"{rel:>8.3%} {label:>13} {arr.mean():>10.5f} {arr.std(ddof=1):>11.6f} "
                  f"{arr.mean() - ref:>10.6f}")

        ratio = np.std(ind, ddof=1) / np.std(crn, ddof=1)
        print(f"{'':>8} {'variance reduction':>13}  {ratio:>8.1f}x lower std dev with CRN\n")


def estimator_comparison() -> None:
    """Accuracy, dispersion and runtime of the three estimators, side by side."""
    ref = bs_greeks(S0, K, R, Q, SIGMA, T, True)
    print("\nestimator comparison (European call, 10 seeds)")
    print(f"{'method':>10} {'delta':>20} {'gamma':>20} {'vega':>20} {'sec/est':>9}")
    print(f"{'':>10} {'mean (sd)':>20} {'mean (sd)':>20} {'mean (sd)':>20}")
    print("-" * 82)
    print(f"{'analytic':>10} {ref.delta:>20.5f} {ref.gamma:>20.5f} {ref.vega:>20.4f} {'-':>9}")

    for method in ("fd", "pathwise", "lr"):
        deltas, gammas, vegas = [], [], []
        t0 = time.perf_counter()
        for seed in range(10):
            g = mc_greeks(S0, K, R, SIGMA, T, True, q=Q, n_paths=N_PATHS, n_steps=N_STEPS,
                          seed=seed, method=method)
            deltas.append(g.delta)
            gammas.append(g.gamma)
            vegas.append(g.vega)
        elapsed = (time.perf_counter() - t0) / 10

        d, gm, v = np.array(deltas), np.array(gammas), np.array(vegas)
        gamma_txt = "n/a (not defined)" if np.all(np.isnan(gm)) else f"{gm.mean():.5f} ({gm.std(ddof=1):.5f})"
        print(f"{method:>10} {f'{d.mean():.5f} ({d.std(ddof=1):.5f})':>20} {gamma_txt:>20} "
              f"{f'{v.mean():.4f} ({v.std(ddof=1):.4f})':>20} {elapsed:>9.3f}")


def american_greeks() -> None:
    """American Greeks via finite differences on LSM, against the European twin."""
    print("\nAmerican vs European put Greeks (S0=95, K=100)")
    print(f"{'style':>10} {'price':>10} {'delta':>10} {'gamma':>10} {'vega':>10}")
    print("-" * 52)
    for style in ("european", "american"):
        g = mc_greeks(95.0, K, R, SIGMA, T, False, q=Q, n_paths=N_PATHS, n_steps=N_STEPS,
                      seed=1, style=style, method="fd")
        print(f"{style:>10} {g.price:>10.4f} {g.delta:>10.4f} {g.gamma:>10.5f} {g.vega:>10.3f}")
    print("\nThe American put is worth more and carries a more negative delta:")
    print("the right to exercise early is most valuable exactly when the option is ITM.")


if __name__ == "__main__":
    crn_vs_independent()
    estimator_comparison()
    american_greeks()
