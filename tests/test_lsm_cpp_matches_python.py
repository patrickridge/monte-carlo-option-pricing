import numpy as np
import pytest

from mcop.simulate_paths import simulate_gbm_paths
from mcop.american_lsm import american_option_lsm

_cpp = pytest.importorskip("mcop._mcop_cpp")
from mcop.american_lsm_cpp import american_option_lsm_cpp

def test_cpp_lsm_matches_python_reasonably():
    S0, K, r, q, sigma, T = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0
    n_steps = 100
    n_paths = 50_000

    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, q=q, seed=123, antithetic=True)

    py_price = american_option_lsm(paths, K, r, T, is_call=False, degree=2, q=q)
    cpp_price = american_option_lsm_cpp(paths, K, r, T, is_call=False, degree=2)

    assert abs(py_price - cpp_price) < 1e-2
