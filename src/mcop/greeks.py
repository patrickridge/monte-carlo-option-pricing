"""
Monte Carlo Greeks: finite-difference, pathwise, and likelihood-ratio estimators.

Greeks are the reason a pricing library becomes a *risk* library. A price alone
tells you what something is worth; the Greeks tell you what happens to that
value when the world moves, which is the only thing a position manager can act
on. Everything in `mcop.portfolio` and `mcop.backtest` is built on the numbers
this module produces.

Three estimators, and when each one is the right tool
-----------------------------------------------------

**1. Finite difference with common random numbers (`method="fd"`)**

The obvious approach — reprice at S+h and S-h, divide by 2h — and the obvious
implementation of it is badly broken. If you re-draw the random numbers for each
bump, the two prices carry independent MC noise of size ~sigma_MC, and the
difference quotient has variance ~2*sigma_MC^2/(2h)^2. As you shrink h to reduce
discretisation bias, the *noise explodes like 1/h^2*. There is no good h.

Common random numbers fix this. Reusing one shock matrix Z for both bumps makes
the two prices strongly positively correlated, and

    Var(A - B) = Var(A) + Var(B) - 2*Cov(A, B)

collapses as Cov -> Var. For a payoff that is smooth in the parameter, the
estimator variance becomes O(1) rather than O(1/h^2), which is the difference
between a usable delta and noise.

`benchmarks/bench_greeks.py` measures this directly, and the measurements show
the predicted scaling: the standard deviation of the delta estimate is **40x
lower** with common random numbers at a 1% bump, **75x** at 0.5%, and **361x**
at 0.1%. The advantage grows as the bump shrinks precisely because the naive
estimator's noise diverges like 1/h while the CRN estimator's does not.

This is the only method here that works for **American** options, because it
treats the pricer as a black box and the LSM exercise policy re-optimises under
each bump.

**2. Pathwise derivative (`method="pathwise"`)**

Differentiate the payoff along each path and average:
dV/dS0 = E[d(payoff)/dS_T * dS_T/dS0]. Unbiased, and typically an order of
magnitude tighter than finite differences because no differencing happens at
all. Requires the payoff to be almost-surely differentiable in the parameter,
which holds for vanilla calls/puts (the kink at K is a measure-zero event) but
**fails for digitals and barriers**, where the payoff jumps.

Cannot produce gamma: the second derivative of a vanilla payoff is a Dirac delta
at the strike, and no finite sample of paths ever lands exactly on it, so the
naive estimator returns exactly zero. That is not a bug to work around — it is
the method telling you it does not apply.

**3. Likelihood ratio (`method="lr"`)**

Differentiate the *probability density* rather than the payoff, and weight each
path by the resulting score function. Because the payoff is never
differentiated, this handles discontinuous payoffs — exactly the case where
pathwise breaks — and it produces gamma without difficulty. The cost is
variance: LR estimators are noisier than pathwise, badly so for short-dated
options where the score function blows up.

Rule of thumb, and what this module defaults to: pathwise where it applies
(delta, vega on vanillas), LR where the payoff misbehaves or you need gamma from
a single pass, finite difference with CRN whenever the pricer is a black box —
which for American exercise is always.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .black_scholes import Greeks
from .simulate_paths import draw_normals, gbm_paths_from_normals
from .american_lsm import american_option_lsm

__all__ = ["mc_greeks"]

# A payoff takes the full path array, shape (n_paths, n_steps + 1), and returns
# one undiscounted payoff per path. Taking whole paths rather than terminal
# prices is what allows path-dependent exotics (Asian, barrier, lookback) to be
# risked with the same machinery as a vanilla.
PayoffFn = Callable[[np.ndarray], np.ndarray]


def _terminal_payoff(ST: np.ndarray, K: float, is_call: bool) -> np.ndarray:
    if is_call:
        return np.maximum(ST - K, 0.0)
    return np.maximum(K - ST, 0.0)


def _price_from_normals(
    Z: np.ndarray,
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    is_call: bool,
    style: str,
    degree: int,
    payoff: PayoffFn | None = None,
) -> float:
    """
    Price a single option from a fixed shock matrix.

    Every finite-difference bump routes through here with the *same* Z, which is
    what makes the differences common-random-number differences.

    `payoff` replaces the vanilla terminal payoff with an arbitrary function of
    the whole path, which is what lets this price and risk the exotics in
    `mcop.exotics`.
    """
    paths = gbm_paths_from_normals(Z, S0=S0, r=r, sigma=sigma, T=T, q=q)

    if style == "american":
        return float(american_option_lsm(paths, K=K, r=r, T=T, is_call=is_call, degree=degree, q=q))

    payoffs = payoff(paths) if payoff is not None else _terminal_payoff(paths[:, -1], K, is_call)
    return float(np.exp(-r * T) * np.asarray(payoffs, dtype=float).mean())


def _greeks_fd(
    Z, S0, K, r, q, sigma, T, is_call, style, degree,
    rel_bump: float, vol_bump: float, t_bump: float, r_bump: float,
    payoff: PayoffFn | None = None,
) -> Greeks:
    """
    Central finite differences under common random numbers.

    Bump sizes are deliberately not infinitesimal. With CRN the variance no
    longer explodes as h shrinks, but discretisation bias and floating-point
    cancellation still trade off, and gamma (a second difference, dividing by
    h^2) is the binding constraint. A 1% relative spot bump is the standard desk
    choice and keeps gamma stable.
    """
    h_S = rel_bump * S0

    def price_at(S0_=S0, sigma_=sigma, T_=T, r_=r):
        return _price_from_normals(Z, S0_, K, r_, q, sigma_, T_, is_call, style, degree, payoff)

    base = price_at()

    up = price_at(S0_=S0 + h_S)
    down = price_at(S0_=S0 - h_S)
    delta = (up - down) / (2.0 * h_S)
    gamma = (up - 2.0 * base + down) / (h_S**2)

    vega = (price_at(sigma_=sigma + vol_bump) - price_at(sigma_=sigma - vol_bump)) / (2.0 * vol_bump)

    # theta is the derivative with respect to *calendar time advancing*, which is
    # the negative of the derivative with respect to time-to-maturity. Clamp the
    # down-bump so a near-expiry option does not ask for negative maturity.
    h_T = min(t_bump, 0.5 * T)
    dV_dT = (price_at(T_=T + h_T) - price_at(T_=T - h_T)) / (2.0 * h_T)
    theta = -dV_dT

    rho = (price_at(r_=r + r_bump) - price_at(r_=r - r_bump)) / (2.0 * r_bump)

    return Greeks(price=base, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def _terminal_normal(paths: np.ndarray, S0: float, r: float, q: float, sigma: float, T: float) -> np.ndarray:
    """
    Recover the aggregated terminal standard normal Z_T from the simulated paths.

    S_T = S0 * exp((r - q - sigma^2/2) T + sigma sqrt(T) Z_T)

    Backing Z_T out of the terminal price (rather than summing the per-step
    shocks) keeps these estimators correct for any step count and avoids
    re-deriving the aggregation.
    """
    return (np.log(paths[:, -1] / S0) - (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _greeks_pathwise(
    Z, S0, K, r, q, sigma, T, is_call, degree,
    vol_bump: float, t_bump: float, r_bump: float,
) -> Greeks:
    """
    Pathwise (infinitesimal perturbation) estimators for delta and vega.

    delta: dV/dS0 = e^{-rT} E[ 1{in the money} * dS_T/dS0 ],  dS_T/dS0 = S_T/S0
    vega : dV/dsigma = e^{-rT} E[ 1{ITM} * dS_T/dsigma ],
           dS_T/dsigma = S_T * (Z_T*sqrt(T) - sigma*T)

    Gamma is not available (see module docstring) and is returned as NaN rather
    than a misleading zero. Theta and rho fall back to finite differences.
    """
    paths = gbm_paths_from_normals(Z, S0=S0, r=r, sigma=sigma, T=T, q=q)
    ST = paths[:, -1]
    disc = np.exp(-r * T)

    payoffs = _terminal_payoff(ST, K, is_call)
    price = float(disc * payoffs.mean())

    # Indicator of finishing in the money, signed by option direction. This is
    # d(payoff)/d(S_T), which exists almost surely.
    if is_call:
        dpayoff_dST = (ST > K).astype(float)
    else:
        dpayoff_dST = -(ST < K).astype(float)

    delta = float(disc * np.mean(dpayoff_dST * (ST / S0)))

    Z_T = _terminal_normal(paths, S0, r, q, sigma, T)
    dST_dsigma = ST * (Z_T * np.sqrt(T) - sigma * T)
    vega = float(disc * np.mean(dpayoff_dST * dST_dsigma))

    # No pathwise gamma for a vanilla payoff.
    gamma = float("nan")

    def price_at(sigma_=sigma, T_=T, r_=r):
        return _price_from_normals(Z, S0, K, r_, q, sigma_, T_, is_call, "european", degree)

    h_T = min(t_bump, 0.5 * T)
    theta = -(price_at(T_=T + h_T) - price_at(T_=T - h_T)) / (2.0 * h_T)
    rho = (price_at(r_=r + r_bump) - price_at(r_=r - r_bump)) / (2.0 * r_bump)

    return Greeks(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def _greeks_lr(
    Z, S0, K, r, q, sigma, T, is_call, degree,
    t_bump: float, r_bump: float,
    payoff: PayoffFn | None = None,
) -> Greeks:
    """
    Likelihood-ratio (score function) estimators.

    Writing the price as an integral of payoff against the lognormal density and
    differentiating the *density* gives weights that multiply an untouched
    payoff. With Z_T the terminal standard normal and v = sigma*sqrt(T):

        delta = e^{-rT} E[ payoff * Z_T / (S0 * v) ]
        gamma = e^{-rT} E[ payoff * (Z_T^2 - 1 - Z_T*v) / (S0^2 * v^2) ]
        vega  = e^{-rT} E[ payoff * ((Z_T^2 - 1)/sigma - Z_T*sqrt(T)) ]

    Note gamma comes out of a single simulation pass with no second differencing
    — the main practical reason to reach for LR on top of its tolerance for
    discontinuous payoffs.
    """
    paths = gbm_paths_from_normals(Z, S0=S0, r=r, sigma=sigma, T=T, q=q)
    ST = paths[:, -1]
    disc = np.exp(-r * T)

    payoffs = payoff(paths) if payoff is not None else _terminal_payoff(ST, K, is_call)
    payoffs = np.asarray(payoffs, dtype=float)
    price = float(disc * payoffs.mean())

    Z_T = _terminal_normal(paths, S0, r, q, sigma, T)
    sqrt_T = np.sqrt(T)
    v = sigma * sqrt_T

    delta = float(disc * np.mean(payoffs * Z_T / (S0 * v)))
    gamma = float(disc * np.mean(payoffs * (Z_T**2 - 1.0 - Z_T * v) / (S0**2 * v**2)))
    vega = float(disc * np.mean(payoffs * ((Z_T**2 - 1.0) / sigma - Z_T * sqrt_T)))

    def price_at(sigma_=sigma, T_=T, r_=r):
        return _price_from_normals(Z, S0, K, r_, q, sigma_, T_, is_call, "european", degree, payoff)

    h_T = min(t_bump, 0.5 * T)
    theta = -(price_at(T_=T + h_T) - price_at(T_=T - h_T)) / (2.0 * h_T)
    rho = (price_at(r_=r + r_bump) - price_at(r_=r - r_bump)) / (2.0 * r_bump)

    return Greeks(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def mc_greeks(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    is_call: bool,
    q: float = 0.0,
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: int | None = 123,
    antithetic: bool = True,
    style: str = "european",
    method: str = "fd",
    degree: int = 2,
    rel_bump: float = 0.01,
    vol_bump: float = 1e-3,
    t_bump: float = 1e-3,
    r_bump: float = 1e-4,
    payoff: PayoffFn | None = None,
) -> Greeks:
    """
    Estimate price and Greeks by Monte Carlo.

    Parameters
    ----------
    style : "european" or "american"
        American exercise is priced by Longstaff–Schwartz and therefore only
        supports method="fd".
    method : "fd", "pathwise", or "lr"
        See the module docstring for the trade-offs.
    payoff : callable, optional
        Custom payoff taking the full path array and returning one undiscounted
        payoff per path — see `mcop.exotics`. Supported by "fd" and "lr", both
        of which need only payoff *values*. Not supported by "pathwise", which
        needs the payoff's derivative and cannot obtain it from an opaque
        callable; that restriction is enforced rather than silently approximated.

        The digital payoff is the case worth trying: it is a step function, so
        pathwise would return exactly zero delta even if it could be applied,
        while "lr" handles it correctly because it never differentiates the
        payoff at all.

    Returns
    -------
    Greeks
        Raw derivatives in the units documented in `mcop.black_scholes`.
        `.scaled()` converts to desk conventions (vega per vol point, theta per
        day).

    Notes
    -----
    A single shock matrix is drawn here and threaded through every bump, so all
    estimates for one call share their randomness. Two calls with the same seed
    are reproducible; two calls with different seeds will differ by MC noise.
    """
    if style not in ("european", "american"):
        raise ValueError("style must be 'european' or 'american'")
    if method not in ("fd", "pathwise", "lr"):
        raise ValueError("method must be 'fd', 'pathwise', or 'lr'")
    if style == "american" and method != "fd":
        raise ValueError(
            "American exercise only supports method='fd'. Pathwise and likelihood-ratio "
            "estimators assume a payoff written on the terminal price; with early exercise "
            "the payoff depends on a stopping rule that is itself estimated."
        )
    if sigma <= 0:
        raise ValueError("sigma must be positive for Greeks (the vol derivative is undefined at 0)")
    if T <= 0:
        raise ValueError("T must be positive")
    if payoff is not None and method == "pathwise":
        raise ValueError(
            "method='pathwise' cannot take a custom payoff: it differentiates the payoff "
            "along each path, which requires knowing that derivative analytically. Use "
            "method='lr' (differentiates the density, so the payoff is never touched) or "
            "method='fd' (treats the payoff as a black box)."
        )
    if payoff is not None and style == "american":
        raise ValueError(
            "custom payoffs are not supported with American exercise: the Longstaff–Schwartz "
            "regression assumes the vanilla intrinsic value defines the exercise decision."
        )

    Z = draw_normals(n_steps=n_steps, n_paths=n_paths, seed=seed, antithetic=antithetic)

    if method == "fd":
        return _greeks_fd(
            Z, S0, K, r, q, sigma, T, is_call, style, degree,
            rel_bump=rel_bump, vol_bump=vol_bump, t_bump=t_bump, r_bump=r_bump,
            payoff=payoff,
        )
    if method == "pathwise":
        return _greeks_pathwise(
            Z, S0, K, r, q, sigma, T, is_call, degree,
            vol_bump=vol_bump, t_bump=t_bump, r_bump=r_bump,
        )
    return _greeks_lr(
        Z, S0, K, r, q, sigma, T, is_call, degree,
        t_bump=t_bump, r_bump=r_bump, payoff=payoff,
    )
