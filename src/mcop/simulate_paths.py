import numpy as np

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
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if T <= 0:
        raise ValueError("T must be positive")

    rng = np.random.default_rng(seed)
    dt = T / n_steps

    m = n_paths
    if antithetic:
        m = (n_paths + 1) // 2

    Z = rng.standard_normal((m, n_steps))
    if antithetic:
        Z = np.vstack([Z, -Z])[:n_paths]

    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_increments = drift + diffusion

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_increments, axis=1))
    return paths
