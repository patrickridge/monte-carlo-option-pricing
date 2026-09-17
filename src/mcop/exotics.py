"""
Exotic option payoffs: Asian, lookback, barrier and digital.

Why these belong in a Monte Carlo library
------------------------------------------
Vanilla options barely need simulation — Black-Scholes prices a European option
in closed form, and a binomial tree handles American exercise. The reason to
build a Monte Carlo engine at all is everything in this module: payoffs that
depend on the *whole path*, not just where it ends up.

That is the honest answer to "why Monte Carlo rather than a tree?". A tree
carries only the current price at each node, so a payoff that depends on the
running average, or the minimum so far, or whether a level was ever touched,
does not fit it without bolting on extra state dimensions and paying the curse
of dimensionality. Monte Carlo already has the full path in memory; the payoff
is a one-line reduction over it.

The four families here, and what each one demands
--------------------------------------------------
- **Asian** (average price). Averaging suppresses volatility, so an Asian is
  always cheaper than the equivalent vanilla. The arithmetic version has no
  closed form; the geometric one does, and is used here both as an exact test
  and as a control variate.
- **Lookback** (best price achieved). Priced off the running maximum or
  minimum. Expensive, because the holder gets the benefit of hindsight.
- **Barrier** (knocked in or out by touching a level). The interesting one
  numerically: a discretely-sampled simulation can *miss* a crossing that
  happened between two time steps, which biases the price. Corrected here with
  a Brownian bridge.
- **Digital** (fixed payout if in the money). Trivial to price, awkward to
  risk-manage: the payoff is a step function, which breaks the pathwise Greek
  estimator and is exactly what the likelihood-ratio estimator exists for.

Fixing convention
-----------------
`simulate_gbm_paths` returns an array whose column 0 is the known starting spot.
Averages and extrema here are taken over columns 1 onward — the *simulated
fixings* — so that n time steps means n fixings at t = T/n, 2T/n, ... , T. This
matches `exotic_analytic.geometric_asian_price` exactly rather than
approximately, which is what lets the geometric price serve as a control
variate. Including the known S0 in an average is a real convention used in some
contracts, but mixing the two silently is a classic source of small, baffling
pricing discrepancies.
"""

from __future__ import annotations

import numpy as np

from .exotic_analytic import geometric_asian_price
from .variance_reduction import control_variate_adjustment

__all__ = [
    "asian_payoff",
    "lookback_payoff",
    "digital_payoff",
    "barrier_payoff",
    "barrier_survival_probability",
    "mc_price_payoff",
    "asian_price_with_control",
]


def _fixings(paths: np.ndarray) -> np.ndarray:
    """Simulated fixings only — column 0 is the known starting spot."""
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("paths must be 2D with at least one simulated step")
    return paths[:, 1:]


def _vanilla(ST: np.ndarray, K: float, is_call: bool) -> np.ndarray:
    return np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)


# --------------------------------------------------------------------------
# Asian
# --------------------------------------------------------------------------

def asian_payoff(
    paths: np.ndarray,
    K: float,
    is_call: bool = True,
    average: str = "arithmetic",
) -> np.ndarray:
    """
    Fixed-strike Asian payoff: max(A - K, 0) for a call, where A is the average
    price over the fixings.

    Arithmetic averaging is the market standard and has no closed form, because
    a sum of lognormal variables is not lognormal. Geometric averaging is
    analytically tractable for exactly that reason — the product of lognormals
    *is* lognormal — and is mostly useful as a benchmark and control variate.

    Asians are common in commodity and FX markets precisely because averaging
    makes the payoff hard to manipulate near expiry and smooths out a single bad
    fixing.
    """
    fx = _fixings(paths)
    if average == "arithmetic":
        A = fx.mean(axis=1)
    elif average == "geometric":
        # Average the logs rather than multiplying: a product of several hundred
        # prices overflows, the sum of their logs does not.
        A = np.exp(np.log(fx).mean(axis=1))
    else:
        raise ValueError("average must be 'arithmetic' or 'geometric'")
    return _vanilla(A, K, is_call)


# --------------------------------------------------------------------------
# Lookback
# --------------------------------------------------------------------------

def lookback_payoff(
    paths: np.ndarray,
    is_call: bool = True,
    K: float | None = None,
) -> np.ndarray:
    """
    Lookback payoff, floating strike when `K is None`, fixed strike otherwise.

    Floating strike — the holder gets the best price in hindsight:
        call: S_T - min(S),  put: max(S) - S_T
    These can never be negative, so a floating lookback always finishes with a
    non-negative payoff. That is why they are expensive.

    Fixed strike — the extreme replaces the terminal price:
        call: max(max(S) - K, 0),  put: max(K - min(S), 0)

    Discrete sampling biases lookbacks *low*, for the same reason it biases
    barriers: the true continuous maximum is at least the sampled maximum, and
    usually higher. The bias shrinks as steps increase, and unlike the barrier
    case there is no cheap bridge correction applied here — worth stating rather
    than leaving implicit.
    """
    fx = _fixings(paths)
    if K is None:
        ST = fx[:, -1]
        return (ST - fx.min(axis=1)) if is_call else (fx.max(axis=1) - ST)
    extreme = fx.max(axis=1) if is_call else fx.min(axis=1)
    return _vanilla(extreme, K, is_call)


# --------------------------------------------------------------------------
# Digital
# --------------------------------------------------------------------------

def digital_payoff(
    paths: np.ndarray,
    K: float,
    is_call: bool = True,
    cash: float = 1.0,
    style: str = "cash",
) -> np.ndarray:
    """
    Digital (binary) payoff on the terminal price.

    cash-or-nothing : pays `cash` if in the money, else 0
    asset-or-nothing: pays S_T if in the money, else 0

    The payoff is a **step function**, and that single fact drives its whole
    numerical character:

    - Pricing is easy — the Monte Carlo estimator is just a discounted
      proportion of paths that finished in the money.
    - Delta is hard. The payoff is flat either side of the strike and jumps in
      between, so its derivative is zero almost everywhere and infinite at one
      point. A pathwise estimator differentiates the payoff and therefore
      returns exactly zero, which is not a small error but a completely wrong
      answer. The likelihood-ratio estimator differentiates the probability
      density instead, never touching the payoff, and handles it correctly.

    This is the payoff that justifies `mcop.greeks` carrying three estimators
    rather than one.
    """
    ST = _fixings(paths)[:, -1]
    itm = (ST > K) if is_call else (ST < K)
    if style == "cash":
        return np.where(itm, float(cash), 0.0)
    if style == "asset":
        return np.where(itm, ST, 0.0)
    raise ValueError("style must be 'cash' or 'asset'")


# --------------------------------------------------------------------------
# Barrier
# --------------------------------------------------------------------------

def barrier_survival_probability(
    paths: np.ndarray,
    H: float,
    direction: str,
    sigma: float,
    T: float,
) -> np.ndarray:
    """
    Probability each path avoided touching the barrier, using a Brownian bridge.

    **The problem this solves.** A simulation observes the price at a finite set
    of times. Between two observations the price moved continuously, and it may
    well have crossed the barrier and come back without either endpoint showing
    it. A naive discrete check therefore under-detects crossings: it prices
    knock-outs too high and knock-ins too low, and the bias does not vanish
    quickly — it decays like 1/sqrt(steps), which is slow.

    **The correction.** Conditional on its two endpoints, a Brownian motion
    between them is a *Brownian bridge*, and the probability that such a bridge
    hit a level has a closed form. For a down-barrier, given the path is above H
    at both ends of a step of length dt:

        P(hit) = exp( -2 * log(S_t/H) * log(S_{t+dt}/H) / (sigma^2 * dt) )

    and symmetrically for an up-barrier with log(H/S). Multiply the
    per-step survival probabilities to get the survival probability for the
    whole path.

    Using the probability directly, rather than drawing a coin flip from it,
    also reduces variance: it replaces a random 0/1 with its conditional
    expectation, which is Rao-Blackwellisation.

    Any step with an endpoint already past the barrier is an outright crossing,
    probability 1.
    """
    if direction not in ("down", "up"):
        raise ValueError("direction must be 'down' or 'up'")
    if sigma <= 0 or T <= 0:
        raise ValueError("sigma and T must be positive")

    n_steps = paths.shape[1] - 1
    dt = T / n_steps
    prev, nxt = paths[:, :-1], paths[:, 1:]

    if direction == "down":
        crossed = (prev <= H) | (nxt <= H)
        safe_prev = np.where(crossed, H * 2.0, prev)  # placeholder keeps log finite
        safe_next = np.where(crossed, H * 2.0, nxt)
        numer = np.log(safe_prev / H) * np.log(safe_next / H)
    else:
        crossed = (prev >= H) | (nxt >= H)
        safe_prev = np.where(crossed, H * 0.5, prev)
        safe_next = np.where(crossed, H * 0.5, nxt)
        numer = np.log(H / safe_prev) * np.log(H / safe_next)

    p_hit = np.exp(-2.0 * numer / (sigma**2 * dt))
    p_hit = np.where(crossed, 1.0, p_hit)
    return np.prod(1.0 - p_hit, axis=1)


def barrier_payoff(
    paths: np.ndarray,
    K: float,
    H: float,
    is_call: bool = True,
    direction: str = "down",
    knock: str = "out",
    sigma: float | None = None,
    T: float | None = None,
    bridge: bool = False,
) -> np.ndarray:
    """
    Barrier option payoff.

    direction : "down" (barrier below spot) or "up" (barrier above)
    knock     : "out" (dies if touched) or "in" (only lives if touched)
    bridge    : apply the Brownian-bridge crossing correction, which needs
                `sigma` and `T`

    With `bridge=False` the barrier is checked only at the simulated times —
    correct for a contract that genuinely monitors discretely (many real ones
    do, e.g. daily close), and biased if the contract monitors continuously.

    With `bridge=True` each path is weighted by its survival probability rather
    than a hard 0/1 indicator, targeting the continuously-monitored price.

    **In-out parity holds either way**: knock-in + knock-out = vanilla, because
    every path either touched or did not. The tests use that identity, since it
    validates the implementation without needing a second closed form.
    """
    if knock not in ("in", "out"):
        raise ValueError("knock must be 'in' or 'out'")

    vanilla = _vanilla(_fixings(paths)[:, -1], K, is_call)

    if bridge:
        if sigma is None or T is None:
            raise ValueError("bridge=True requires sigma and T")
        survival = barrier_survival_probability(paths, H, direction, sigma, T)
    else:
        if direction == "down":
            touched = (paths[:, 1:] <= H).any(axis=1)
        elif direction == "up":
            touched = (paths[:, 1:] >= H).any(axis=1)
        else:
            raise ValueError("direction must be 'down' or 'up'")
        survival = (~touched).astype(float)

    return vanilla * (survival if knock == "out" else (1.0 - survival))


# --------------------------------------------------------------------------
# Pricing helpers
# --------------------------------------------------------------------------

def mc_price_payoff(payoffs: np.ndarray, r: float, T: float) -> tuple[float, float, float, float]:
    """
    Discount a payoff vector and report price, standard error and 95% interval.

    A thin wrapper over `mcop.pricing.mc_price`, re-exported here so exotic
    pricing reads as one import.
    """
    from .pricing import mc_price

    return mc_price(payoffs, r, T)


def asian_price_with_control(
    paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    sigma: float,
    is_call: bool = True,
    q: float = 0.0,
) -> tuple[float, float]:
    """
    Price an arithmetic Asian using the geometric Asian as a control variate.

    **The idea.** The arithmetic and geometric averages of the same path are
    almost perfectly correlated. The geometric option's true price is known
    exactly, so the error the simulation makes on the geometric payoff is
    measurable — and because the two are so correlated, that measured error
    predicts most of the error on the arithmetic payoff. Subtract it off:

        estimate = mean(arith) - beta * (mean(geo) - true_geo)

    with beta the regression coefficient of one on the other. The estimator
    stays unbiased because E[geo - true_geo] = 0 by construction.

    This is the textbook application of control variates, and it works here
    precisely because the correlation is so high — typically above 0.99, which
    is where the technique earns large variance reductions rather than token
    ones.

    Returns
    -------
    (price, beta) : the controlled price estimate and the fitted coefficient.
    """
    n_fixings = paths.shape[1] - 1
    S0 = float(paths[0, 0])
    disc = np.exp(-r * T)

    arith = disc * asian_payoff(paths, K, is_call, "arithmetic")
    geo = disc * asian_payoff(paths, K, is_call, "geometric")
    geo_true = geometric_asian_price(S0, K, r, q, sigma, T, n_fixings, is_call)

    return control_variate_adjustment(arith, geo, geo_true)
