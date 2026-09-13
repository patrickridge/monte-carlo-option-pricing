import numpy as np


def draw_normals(
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = False,
) -> np.ndarray:
    """
    Draw the standard normal shocks that drive a GBM simulation.

    Splitting this out from path construction is what makes **common random
    numbers** possible: the same shock matrix can be reused to build paths under
    bumped parameters, so a finite-difference Greek differences two paths driven
    by identical randomness. Without that, the MC noise in each leg of the bump
    swamps the signal — see `mcop.greeks` for the variance argument.

    Returns
    -------
    Z : ndarray, shape (n_paths, n_steps)
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    rng = np.random.default_rng(seed)

    m = n_paths
    if antithetic:
        m = (n_paths + 1) // 2

    Z = rng.standard_normal((m, n_steps))
    if antithetic:
        # Pair each draw with its negation: the average of an antithetic pair has
        # the same mean but lower variance for payoffs monotone in the shock.
        Z = np.vstack([Z, -Z])[:n_paths]
    return Z


def gbm_paths_from_normals(
    Z: np.ndarray,
    S0: float,
    r: float,
    sigma: float,
    T: float,
    q: float = 0.0,
) -> np.ndarray:
    """
    Build GBM paths from a pre-drawn shock matrix.

    Exact discretization (no Euler bias):
    S_{t+dt} = S_t * exp((r-q-0.5*sigma^2)dt + sigma*sqrt(dt)*Z)

    Returns
    -------
    paths : ndarray, shape (n_paths, n_steps + 1) with paths[:, 0] = S0
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if T <= 0:
        raise ValueError("T must be positive")

    n_paths, n_steps = Z.shape
    dt = T / n_steps

    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_increments = drift + diffusion

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_increments, axis=1))
    return paths


def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    q: float = 0.0,
    seed: int | None = None,
    antithetic: bool = False,
) -> np.ndarray:
    """
    Simulate GBM paths under the risk-neutral measure.

    dS_t = (r - q) S_t dt + sigma S_t dW_t

    Exact discretization:
    S_{t+dt} = S_t * exp((r-q-0.5*sigma^2)dt + sigma*sqrt(dt)*Z),  Z~N(0,1)

    Returns
    -------
    paths : ndarray, shape (n_paths, n_steps + 1)
        paths[:, 0] = S0
    """
    Z = draw_normals(n_steps=n_steps, n_paths=n_paths, seed=seed, antithetic=antithetic)
    return gbm_paths_from_normals(Z, S0=S0, r=r, sigma=sigma, T=T, q=q)
