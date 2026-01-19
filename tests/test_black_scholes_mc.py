import math
import numpy as np

from mcop.simulate_paths import simulate_gbm_paths
from mcop.payoffs import european_call, european_put
from mcop.pricing import mc_price

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black_scholes_call_put(S0, K, r, q, sigma, T):
    if T <= 0:
        raise ValueError("T must be > 0")
    if sigma <= 0:
        # Degenerate: deterministic under RN
        fwd = S0 * math.exp((r - q) * T)
        call = math.exp(-r * T) * max(fwd - K, 0.0)
        put  = math.exp(-r * T) * max(K - fwd, 0.0)
        return call, put

    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call = S0 * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    put  = K * math.exp(-r * T) * _norm_cdf(-d2) - S0 * math.exp(-q * T) * _norm_cdf(-d1)
    return call, put

def test_mc_matches_black_scholes_reasonably():
    # modest params
    S0 = 100.0
    K = 100.0
    r = 0.05
    q = 0.00
    sigma = 0.2
    T = 1.0
    n_steps = 252
    n_paths = 200_000

    bs_call, bs_put = black_scholes_call_put(S0, K, r, q, sigma, T)

    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, q=q, seed=123, antithetic=True)
    call_payoffs = european_call(paths, K)
    put_payoffs = european_put(paths, K)

    mc_call, se_call, lo_c, hi_c = mc_price(call_payoffs, r, T)
    mc_put,  se_put,  lo_p, hi_p = mc_price(put_payoffs, r, T)

    # Check BS lies within MC CI (usually true at this N)
    assert lo_c <= bs_call <= hi_c
    assert lo_p <= bs_put <= hi_p
