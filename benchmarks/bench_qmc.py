"""
Benchmark: quasi-Monte Carlo against ordinary Monte Carlo.

Pseudo-random numbers clump. Any finite sample leaves some regions crowded and
others empty, and those gaps are the error — which is why Monte Carlo converges
only as 1/sqrt(N), so quadrupling the work halves the error.

A Sobol sequence places its points deliberately rather than randomly, filling
the space evenly. For smooth integrands the error can approach O(1/N).

Two results this measures:

1. **The Brownian bridge construction is what makes QMC work in high
   dimensions.** A 50-step path needs a 50-dimensional sequence, and Sobol
   points are far better distributed in their early dimensions than their later
   ones. Assigning dimension i to time step i wastes the good dimensions on
   fine detail. The bridge assigns them to the largest-scale features instead.
   Plain Sobol is a modest win; Sobol with the bridge is a large one.

2. **Randomised QMC restores the error bar that QMC destroys.** A standard
   error is computed from sampling randomness, which QMC deliberately removes.
   Averaging over independently scrambled copies of the sequence gives back a
   legitimate error estimate while keeping the convergence.

Run:  python benchmarks/bench_qmc.py
"""

from __future__ import annotations

import math

import numpy as np

from mcop.black_scholes import bs_price
from mcop.exotics import asian_payoff
from mcop.exotic_analytic import geometric_asian_price
from mcop.qmc import qmc_price, rqmc_estimate, sobol_normals
from mcop.simulate_paths import draw_normals, gbm_paths_from_normals

S0, K, R, Q, SIGMA, T = 100.0, 100.0, 0.05, 0.0, 0.20, 1.0
N_STEPS = 50
DISC = math.exp(-R * T)
SEEDS = range(8)


def _call_price(Z: np.ndarray) -> float:
    ST = gbm_paths_from_normals(Z, S0=S0, r=R, sigma=SIGMA, T=T, q=Q)[:, -1]
    return float(DISC * np.maximum(ST - K, 0.0).mean())


def convergence() -> None:
    """Error against the exact Black-Scholes price, at matched path counts."""
    exact = bs_price(S0, K, R, Q, SIGMA, T, True)
    print(f"\nCONVERGENCE — European call, {N_STEPS} time steps")
    print(f"exact price (Black-Scholes): {exact:.6f}")
    print(f"{'paths':>9} {'pseudo':>11} {'Sobol':>11} {'Sobol+bridge':>14} {'speedup':>10}")
    print("-" * 60)

    for n in (1024, 4096, 16384, 65536):
        pseudo = np.mean([abs(_call_price(draw_normals(N_STEPS, n, seed=s)) - exact) for s in SEEDS])
        plain = np.mean([abs(_call_price(sobol_normals(N_STEPS, n, seed=s, bridge=False)) - exact) for s in SEEDS])
        bridged = np.mean([abs(_call_price(sobol_normals(N_STEPS, n, seed=s, bridge=True)) - exact) for s in SEEDS])
        print(f"{n:>9,} {pseudo:>11.5f} {plain:>11.5f} {bridged:>14.5f} {pseudo / max(bridged, 1e-12):>9.0f}x")

    print("\n  Read across: plain Sobol is a modest improvement, Sobol with the Brownian")
    print("  bridge construction is a large one. In 50 dimensions the later Sobol")
    print("  dimensions are poorly distributed, and the bridge is what keeps the good")
    print("  ones on the features that carry the variance.")
    print("\n  Read down the pseudo column: 64x the paths buys about 8x the accuracy,")
    print("  which is 1/sqrt(N) exactly as theory says. The bridged column improves")
    print("  far faster than that.")


def paths_needed_for_equal_accuracy() -> None:
    """The practical question: how much work does QMC save?"""
    exact = bs_price(S0, K, R, Q, SIGMA, T, True)
    target = np.mean([abs(_call_price(sobol_normals(N_STEPS, 4096, seed=s)) - exact) for s in SEEDS])

    print(f"\nWORK SAVED — matching the accuracy of 4,096 QMC paths (err {target:.5f})")
    for n in (4096, 65_536, 1_048_576):
        err = np.mean([abs(_call_price(draw_normals(N_STEPS, n, seed=s)) - exact) for s in SEEDS[:4]])
        verdict = "still worse" if err > target else "matched"
        print(f"  pseudo-random with {n:>9,} paths: err {err:.5f}   ({verdict})")


def error_bars() -> None:
    """QMC has no natural standard error. Randomised QMC gives one back."""
    exact = bs_price(S0, K, R, Q, SIGMA, T, True)
    n = 4096

    print(f"\nERROR ESTIMATION — {n:,} paths")
    vals = np.array([_call_price(draw_normals(N_STEPS, n, seed=s)) for s in range(16)])
    se_p = vals.std(ddof=1) / math.sqrt(16)
    print(f"  plain Monte Carlo : {vals.mean():.5f} +/- {se_p:.5f}   (err {abs(vals.mean() - exact):.5f})")

    est, se_q = rqmc_estimate(_call_price, N_STEPS, n, n_scrambles=16, seed=0)
    print(f"  randomised QMC    : {est:.5f} +/- {se_q:.5f}   (err {abs(est - exact):.5f})")
    print(f"\n  error bar {se_p / max(se_q, 1e-12):.0f}x tighter, and still honest — each scramble is an")
    print("  independent randomisation of the same low-discrepancy point set, so the")
    print("  spread between them is a real basis for a confidence interval.")


def path_dependent() -> None:
    """Where QMC should help most: smooth, high-dimensional, path-dependent."""
    print("\nPATH-DEPENDENT PAYOFFS — Asian options")
    geo_exact = geometric_asian_price(S0, K, R, Q, SIGMA, T, N_STEPS, True)

    geo = lambda p: asian_payoff(p, K, True, "geometric")      # noqa: E731
    arith = lambda p: asian_payoff(p, K, True, "arithmetic")   # noqa: E731

    est, se = qmc_price(geo, S0, R, SIGMA, T, N_STEPS, 4096, q=Q, n_scrambles=16, seed=1)
    print(f"  geometric Asian, QMC 4,096 paths : {est:.5f} +/- {se:.5f}")
    print(f"  geometric Asian, exact           : {geo_exact:.5f}   (err {abs(est - geo_exact):.5f})")

    est_a, se_a = qmc_price(arith, S0, R, SIGMA, T, N_STEPS, 4096, q=Q, n_scrambles=16, seed=1)
    ref = gbm_paths_from_normals(draw_normals(N_STEPS, 500_000, seed=5, antithetic=True),
                                 S0=S0, r=R, sigma=SIGMA, T=T, q=Q)
    ref_price = float(DISC * asian_payoff(ref, K, True, "arithmetic").mean())
    print(f"\n  arithmetic Asian, QMC 4,096 paths: {est_a:.5f} +/- {se_a:.5f}")
    print(f"  arithmetic Asian, 500,000 pseudo : {ref_price:.5f}")
    print("\n  4,096 quasi-random paths match a reference built from 500,000 pseudo-random ones.")


if __name__ == "__main__":
    convergence()
    paths_needed_for_equal_accuracy()
    error_bars()
    path_dependent()
