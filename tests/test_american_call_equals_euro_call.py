import math
import numpy as np

from mcop.simulate_paths import simulate_gbm_paths
from mcop.american_lsm import american_option_lsm
from mcop.payoffs import european_call
from mcop.pricing import mc_price

def test_american_call_matches_european_no_dividends():
    S0 = 100.0
    K = 100.0
    r = 0.05
    q = 0.0
    sigma = 0.2
    T = 1.0

    n_steps = 100
    n_paths = 200_000

    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, q=q, seed=7, antithetic=True)

    euro_pay = european_call(paths, K)
    euro_price, _, lo, hi = mc_price(euro_pay, r, T)

    amer_price = american_option_lsm(paths, K, r, T, is_call=True, degree=2, q=q)

    # American call should be very close; allow small MC noise
    assert lo - 0.10 <= amer_price <= hi + 0.10
