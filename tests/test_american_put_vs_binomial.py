import numpy as np

from mcop.simulate_paths import simulate_gbm_paths
from mcop.american_lsm import american_option_lsm
from mcop.binomial_tree import american_option_crr

def test_american_put_lsm_close_to_binomial():
    S0 = 100.0
    K = 100.0
    r = 0.05
    q = 0.0
    sigma = 0.2
    T = 1.0

    n_steps = 100
    n_paths = 200_000

    # binomial reference (increase steps for accuracy)
    ref = american_option_crr(S0, K, r, sigma, T, n_steps=500, is_call=False, q=q)

    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, q=q, seed=123, antithetic=True)
    est = american_option_lsm(paths, K, r, T, is_call=False, degree=2, q=q)

    # LSM has MC noise + regression bias; use a sensible tolerance
    assert abs(est - ref) < 0.25
