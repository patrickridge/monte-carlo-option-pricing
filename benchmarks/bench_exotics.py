"""
Benchmark: exotic payoffs, and the two numerical techniques they motivate.

Three results worth quoting:

1. **The Brownian bridge fixes barrier discretisation bias.** A discretely
   sampled simulation misses barrier crossings that happen between observation
   times, pricing knock-outs too high. The bias decays like 1/sqrt(steps), which
   is slow enough that brute force is not a fix. The bridge correction is more
   accurate at 25 steps than naive sampling is at 200.

2. **Control variates work when correlation is near 1.** The geometric-average
   Asian has a closed form and correlates above 0.999 with the arithmetic one,
   which has none. Using it as a control cuts the standard error several-fold
   for essentially no extra compute.

3. **Digitals separate the Greek estimators.** A step payoff is where the
   pathwise method stops applying and the likelihood-ratio method earns its
   keep.

Run:  python benchmarks/bench_exotics.py
"""

from __future__ import annotations

import math

import numpy as np

from mcop.black_scholes import bs_price
from mcop.exotic_analytic import (
    digital_cash_or_nothing,
    down_and_out_call,
    geometric_asian_price,
)
from mcop.exotics import (
    asian_payoff,
    barrier_payoff,
    digital_payoff,
    lookback_payoff,
)
from mcop.greeks import mc_greeks
from mcop.simulate_paths import simulate_gbm_paths

S0, K, R, Q, SIGMA, T = 100.0, 100.0, 0.05, 0.0, 0.20, 1.0
N_PATHS = 200_000
DISC = math.exp(-R * T)


def _price(payoffs) -> float:
    return float(DISC * np.asarray(payoffs, dtype=float).mean())


def barrier_bridge() -> None:
    """How much discretisation bias the bridge correction removes."""
    H = 90.0
    reference = down_and_out_call(S0, K, H, R, Q, SIGMA, T)

    print("\nBARRIER — down-and-out call, barrier 90, vs continuous-monitoring closed form")
    print(f"reference (continuous monitoring): {reference:.5f}")
    print(f"{'steps':>7} {'naive':>10} {'error':>10} {'bridged':>10} {'error':>10} {'improvement':>13}")
    print("-" * 64)

    for steps in (25, 50, 100, 200, 400):
        paths = simulate_gbm_paths(S0, R, SIGMA, T, steps, N_PATHS, q=Q, seed=7, antithetic=True)
        naive = _price(barrier_payoff(paths, K, H, True, "down", "out"))
        bridged = _price(barrier_payoff(paths, K, H, True, "down", "out",
                                        sigma=SIGMA, T=T, bridge=True))
        e_naive, e_bridge = naive - reference, bridged - reference
        ratio = abs(e_naive) / max(abs(e_bridge), 1e-12)
        print(f"{steps:>7} {naive:>10.5f} {e_naive:>+10.5f} {bridged:>10.5f} {e_bridge:>+10.5f} {ratio:>12.1f}x")

    print("\n  The naive error decays like 1/sqrt(steps) — 16x the work between the first")
    print("  and last row buys only about a 4x error reduction. The bridge is more accurate")
    print("  at 25 steps than naive sampling is at 400.")


def asian_control_variate() -> None:
    """How much variance the geometric control variate removes."""
    print("\nASIAN — arithmetic average call, plain Monte Carlo vs geometric control variate")
    print(f"{'paths':>9} {'plain':>10} {'std err':>10} {'controlled':>12} {'std err':>10} {'reduction':>11}")
    print("-" * 66)

    for n in (25_000, 50_000, 100_000, 200_000):
        paths = simulate_gbm_paths(S0, R, SIGMA, T, 50, n, q=Q, seed=3, antithetic=True)
        arith = DISC * asian_payoff(paths, K, True, "arithmetic")
        geo = DISC * asian_payoff(paths, K, True, "geometric")
        geo_true = geometric_asian_price(S0, K, R, Q, SIGMA, T, 50, True)

        beta = np.cov(arith, geo, ddof=1)[0, 1] / np.var(geo, ddof=1)
        controlled = arith - beta * (geo - geo_true)

        se_plain = arith.std(ddof=1) / math.sqrt(n)
        se_ctrl = controlled.std(ddof=1) / math.sqrt(n)
        print(f"{n:>9,} {arith.mean():>10.5f} {se_plain:>10.5f} {controlled.mean():>12.5f} "
              f"{se_ctrl:>10.5f} {se_plain / se_ctrl:>10.1f}x")

    paths = simulate_gbm_paths(S0, R, SIGMA, T, 50, N_PATHS, q=Q, seed=3, antithetic=True)
    arith = asian_payoff(paths, K, True, "arithmetic")
    geo = asian_payoff(paths, K, True, "geometric")
    print(f"\n  correlation(arithmetic, geometric) = {np.corrcoef(arith, geo)[0, 1]:.6f}")
    print("  A control variate's benefit scales with that correlation; at 0.999 it is large.")


def digital_greeks() -> None:
    """Which Greek estimators survive a discontinuous payoff."""
    d2 = (math.log(S0 / K) + (R - Q - 0.5 * SIGMA**2) * T) / (SIGMA * math.sqrt(T))
    phi = math.exp(-0.5 * d2 * d2) / math.sqrt(2.0 * math.pi)
    ref_delta = math.exp(-R * T) * phi / (S0 * SIGMA * math.sqrt(T))
    ref_price = digital_cash_or_nothing(S0, K, R, Q, SIGMA, T, True)

    print("\nDIGITAL — cash-or-nothing call, delta by each estimator (10 seeds)")
    print(f"{'method':>12} {'delta mean':>12} {'std dev':>10} {'vs closed form':>16}")
    print("-" * 54)
    print(f"{'closed form':>12} {ref_delta:>12.5f} {'-':>10} {'-':>16}")

    payoff = lambda p: digital_payoff(p, K, True, cash=1.0)  # noqa: E731

    for method in ("lr", "fd"):
        vals = [
            mc_greeks(S0, K, R, SIGMA, T, True, q=Q, n_paths=200_000, n_steps=25,
                      seed=s, method=method, rel_bump=0.02, payoff=payoff).delta
            for s in range(10)
        ]
        a = np.array(vals)
        print(f"{method:>12} {a.mean():>12.5f} {a.std(ddof=1):>10.5f} {a.mean() - ref_delta:>+16.5f}")

    print(f"\n  price check: closed form {ref_price:.5f}")
    print("  pathwise is absent because it cannot be applied at all: it differentiates the")
    print("  payoff, and a step function's derivative is zero everywhere except one point")
    print("  no sample ever lands on. The library raises rather than returning a false zero.")


def exotic_price_table() -> None:
    """Where each exotic sits relative to the vanilla, and why."""
    paths = simulate_gbm_paths(S0, R, SIGMA, T, 50, N_PATHS, q=Q, seed=7, antithetic=True)
    vanilla = bs_price(S0, K, R, Q, SIGMA, T, True)

    rows = [
        ("vanilla call", vanilla, "reference"),
        ("Asian (arithmetic)", _price(asian_payoff(paths, K, True, "arithmetic")),
         "cheaper — averaging damps volatility"),
        ("Asian (geometric)", _price(asian_payoff(paths, K, True, "geometric")),
         "cheaper still — geometric mean <= arithmetic"),
        ("down-and-out (H=90)", _price(barrier_payoff(paths, K, 90.0, True, "down", "out",
                                                      sigma=SIGMA, T=T, bridge=True)),
         "cheaper — can be knocked out"),
        ("down-and-in (H=90)", _price(barrier_payoff(paths, K, 90.0, True, "down", "in",
                                                     sigma=SIGMA, T=T, bridge=True)),
         "the complement; in + out = vanilla"),
        ("lookback (floating)", _price(lookback_payoff(paths, True)),
         "dearer — buys at the minimum, in hindsight"),
        ("digital (cash, x100)", 100.0 * _price(digital_payoff(paths, K, True)),
         "a discounted bet on finishing in the money"),
    ]

    print("\nEXOTIC PRICE COMPARISON — S0=100, K=100, sigma=20%, T=1y")
    print(f"{'payoff':>22} {'price':>10}   rationale")
    print("-" * 74)
    for name, value, why in rows:
        print(f"{name:>22} {value:>10.4f}   {why}")


if __name__ == "__main__":
    exotic_price_table()
    barrier_bridge()
    asian_control_variate()
    digital_greeks()
