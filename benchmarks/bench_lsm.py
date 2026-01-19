import time
import csv
from pathlib import Path

import numpy as np

from mcop.simulate_paths import simulate_gbm_paths
from mcop.american_lsm import american_option_lsm
from mcop.binomial_tree import american_option_crr

OUT = Path("artifacts/bench_lsm.csv")

def main():
    S0, K, r, q, sigma, T = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0
    n_steps = 100
    seed = 123

    # reference (binomial)
    ref = american_option_crr(S0, K, r, sigma, T, n_steps=1000, is_call=False, q=q)

    grid = [
        (10_000, 2), (10_000, 3),
        (50_000, 2), (50_000, 3),
        (100_000, 2), (100_000, 3),
        (200_000, 2), (200_000, 3),
    ]

    rows = []
    for n_paths, degree in grid:
        t0 = time.perf_counter()
        paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, q=q, seed=seed, antithetic=True)
        sim_t = time.perf_counter()

        est = american_option_lsm(paths, K, r, T, is_call=False, degree=degree, q=q)
        t1 = time.perf_counter()

        rows.append({
            "n_paths": n_paths,
            "n_steps": n_steps,
            "degree": degree,
            "ref": ref,
            "est": est,
            "abs_err": abs(est - ref),
            "sim_s": sim_t - t0,
            "lsm_s": t1 - sim_t,
            "total_s": t1 - t0,
        })
        print(rows[-1])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved: {OUT}")

if __name__ == "__main__":
    main()
