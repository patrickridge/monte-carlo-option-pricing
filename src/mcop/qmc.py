"""
Quasi-Monte Carlo: Sobol sequences, scrambling, and Brownian bridge construction.

Ordinary Monte Carlo draws pseudo-random numbers, and pseudo-random numbers
**clump**. Purely by chance any finite sample leaves some regions of the space
crowded and others empty, and those gaps are the error. That is why the error
shrinks only as 1/sqrt(N): quadrupling the work halves the error.

Quasi-Monte Carlo replaces randomness with a deterministic **low-discrepancy
sequence** — points deliberately placed to fill the space as evenly as
possible. No clumps, no gaps. For smooth integrands the error can approach
O(1/N) rather than O(1/sqrt(N)), which for the path counts used here is a large
difference.

Two problems have to be solved to make that work in practice, and both are
handled below.

**Problem 1: QMC destroys your error bars.**
A standard error is computed *from* the randomness of the sample. Remove the
randomness and there is nothing left to compute it from — you get a number with
no honest confidence interval, which for pricing is unacceptable. The fix is
**randomised QMC**: take several independently *scrambled* copies of the same
Sobol sequence. Each copy is still low-discrepancy (so you keep the convergence
gain), but the scrambles differ, so the spread across them is a legitimate error
estimate. `rqmc_estimate` does exactly this.

**Problem 2: QMC degrades in high dimensions.**
A path with 100 time steps needs a 100-dimensional sequence, and Sobol points
are far better distributed in their early dimensions than their later ones. The
standard remedy is the **Brownian bridge construction**: instead of using
dimension *i* for time step *i*, use the first dimension to fix the terminal
value, the second to fix the midpoint, the fourth and third to fix the quarter
points, and so on by bisection. The best-behaved dimensions then control the
largest-scale features of the path, where almost all of the variance lives, and
the poorly-behaved tail dimensions only nudge fine detail.

Note this is the same Brownian bridge that `mcop.exotics` uses to correct
barrier crossings — the same mathematical object (a Brownian motion conditioned
on its endpoints), applied to a completely different problem.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri

from .simulate_paths import gbm_paths_from_normals

__all__ = [
    "sobol_normals",
    "bridge_order",
    "rqmc_estimate",
    "qmc_price",
]

# Sobol points come in powers of two. Asking for a different count silently
# damages the sequence's balance properties, so the helpers below round up and
# say so rather than quietly returning a worse sequence.
_EPS = 1e-12


def _next_power_of_two(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def bridge_order(n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the Brownian bridge construction schedule for `n_steps` time points.

    Returns four arrays of length n_steps describing, for each Sobol dimension
    in order, which time index it sets and how:

        idx[k]   : the time index filled at step k
        left[k]  : index of the known point to its left  (-1 means time 0)
        right[k] : index of the known point to its right (-1 means "none yet",
                   i.e. this is the terminal point)
        sd[k]    : standard deviation of the conditional draw

    The first entry always fills the terminal point, because that is where most
    of the variance of a path lives and it therefore deserves the best-behaved
    Sobol dimension.
    """
    idx = np.zeros(n_steps, dtype=int)
    left = np.zeros(n_steps, dtype=int)
    right = np.zeros(n_steps, dtype=int)
    sd = np.zeros(n_steps, dtype=float)

    filled = np.zeros(n_steps, dtype=bool)

    # Dimension 0 sets the endpoint: W(T) ~ N(0, T) with T = n_steps in step units.
    idx[0] = n_steps - 1
    left[0] = -1
    right[0] = -1
    sd[0] = np.sqrt(n_steps)
    filled[n_steps - 1] = True

    k = 1
    # Repeatedly bisect the largest unfilled gap, mirroring the classic
    # construction. A simple queue of (lo, hi) exclusive-bounds intervals gives
    # the standard ordering.
    intervals = [(-1, n_steps - 1)]
    while k < n_steps:
        new_intervals = []
        for lo, hi in intervals:
            if hi - lo <= 1:
                continue
            mid = (lo + hi) // 2
            if lo < 0:
                # Left endpoint is time 0, which is known and equal to zero.
                span_l = mid + 1
            else:
                span_l = mid - lo
            span_r = hi - mid

            idx[k] = mid
            left[k] = lo
            right[k] = hi
            sd[k] = np.sqrt(span_l * span_r / (span_l + span_r))
            filled[mid] = True
            k += 1

            new_intervals.append((lo, mid))
            new_intervals.append((mid, hi))
            if k >= n_steps:
                break
        if not new_intervals:
            break
        intervals = new_intervals

    return idx, left, right, sd


def _bridge_from_normals(Z: np.ndarray) -> np.ndarray:
    """
    Turn a (n_paths, n_steps) array of independent normals into per-step
    increments, using the Brownian bridge construction.

    Returns per-step standardised increments, so the output is a drop-in
    replacement for the shocks `gbm_paths_from_normals` expects. The marginal
    distribution is identical to the standard construction — only the mapping
    from input dimensions to path features changes, which is exactly what makes
    it useful for QMC and irrelevant for pseudo-random draws.
    """
    n_paths, n_steps = Z.shape
    idx, left, right, sd = bridge_order(n_steps)

    W = np.zeros((n_paths, n_steps), dtype=float)

    for k in range(n_steps):
        i, lo, hi = idx[k], left[k], right[k]
        if hi < 0:
            # Terminal point, conditioned only on W(0) = 0.
            W[:, i] = sd[k] * Z[:, k]
            continue

        w_left = np.zeros(n_paths) if lo < 0 else W[:, lo]
        span_l = (i + 1) if lo < 0 else (i - lo)
        span_r = hi - i
        weight = span_l / (span_l + span_r)
        W[:, i] = w_left + weight * (W[:, hi] - w_left) + sd[k] * Z[:, k]

    # Convert the level process W to standardised per-step increments.
    increments = np.diff(W, axis=1, prepend=0.0)
    return increments  # already unit-variance per step, since sd used step units


def sobol_normals(
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    scramble: bool = True,
    bridge: bool = True,
) -> np.ndarray:
    """
    Standard normal shocks from a scrambled Sobol sequence.

    Drop-in replacement for `simulate_paths.draw_normals`, so anything that
    accepts a shock matrix works unchanged.

    Parameters
    ----------
    scramble : bool
        Apply Owen-style scrambling. Keep this on: an unscrambled Sobol sequence
        is fully deterministic, which removes any basis for an error estimate and
        can also produce visible artefacts in low dimensions.
    bridge : bool
        Use the Brownian bridge construction so the best Sobol dimensions
        control the largest-scale path features. Matters increasingly as
        `n_steps` grows; at 100 steps it is the difference between QMC helping
        and QMC doing nothing.

    Notes
    -----
    `n_paths` is rounded **up** to a power of two. Sobol sequences are balanced
    in blocks of 2^m, and taking some other count quietly degrades the very
    uniformity the method exists to provide.
    """
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be positive")

    n = _next_power_of_two(n_paths)
    m = int(np.log2(n))

    sampler = qmc.Sobol(d=n_steps, scramble=scramble, seed=seed)
    u = sampler.random_base2(m)

    # ndtri maps (0,1) -> R and diverges at the endpoints. Scrambled Sobol can
    # legitimately produce a 0, so clip rather than emit -inf.
    u = np.clip(u, _EPS, 1.0 - _EPS)
    Z = ndtri(u)

    if bridge:
        Z = _bridge_from_normals(Z)
    return Z[:n_paths] if n_paths < n else Z


def rqmc_estimate(
    price_fn,
    n_steps: int,
    n_paths: int,
    n_scrambles: int = 16,
    seed: int | None = 0,
    bridge: bool = True,
) -> tuple[float, float]:
    """
    Randomised QMC: average over independently scrambled Sobol sequences, and
    use the spread between them as an honest error estimate.

    `price_fn` takes a shock matrix and returns a scalar price.

    Why this exists: plain QMC gives a number with no confidence interval,
    because the standard error is computed from sampling randomness that QMC has
    deliberately removed. Each scramble here is an independent randomisation of
    the *same* low-discrepancy point set, so every replicate keeps the
    convergence benefit while the variation across replicates is a legitimate
    basis for an error bar.

    Returns
    -------
    (estimate, standard_error)
        Mean across scrambles, and the standard error of that mean.
    """
    if n_scrambles < 2:
        raise ValueError("need at least 2 scrambles to estimate an error")

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_scrambles)

    values = np.array([
        price_fn(sobol_normals(n_steps, n_paths, seed=int(s), scramble=True, bridge=bridge))
        for s in seeds
    ], dtype=float)

    est = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n_scrambles))
    return est, se


def qmc_price(
    payoff,
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    q: float = 0.0,
    n_scrambles: int = 16,
    seed: int | None = 0,
    bridge: bool = True,
) -> tuple[float, float]:
    """
    Price any payoff by randomised quasi-Monte Carlo.

    `payoff` takes the full path array and returns one undiscounted payoff per
    path, matching the convention in `mcop.exotics`.

    Returns (price, standard_error).
    """
    disc = np.exp(-r * T)

    def price_fn(Z: np.ndarray) -> float:
        paths = gbm_paths_from_normals(Z, S0=S0, r=r, sigma=sigma, T=T, q=q)
        return float(disc * np.asarray(payoff(paths), dtype=float).mean())

    return rqmc_estimate(price_fn, n_steps, n_paths, n_scrambles, seed, bridge)
