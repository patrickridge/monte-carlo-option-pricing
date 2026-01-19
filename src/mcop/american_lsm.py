import numpy as np

def _basis_poly(S: np.ndarray, degree: int) -> np.ndarray:
    """
    Polynomial basis [1, S, S^2, ..., S^degree].
    Returns shape (n, degree+1).
    """
    cols = [np.ones_like(S)]
    for k in range(1, degree + 1):
        cols.append(S ** k)
    return np.column_stack(cols)

def american_option_lsm(
    paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    is_call: bool,
    degree: int = 2,
    q: float = 0.0,
    return_cashflows: bool = False,
) -> float | tuple[float, np.ndarray]:
    """
    Longstaff–Schwartz Monte Carlo pricer for American options.

    If return_cashflows=True, returns (price, discounted_cashflows_per_path),
    where discounted_cashflows_per_path are discounted to time 0.
    """
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    dt = T / n_steps
    disc = np.exp(-r * dt)

    # payoff at a given spot
    def payoff(S):
        if is_call:
            return np.maximum(S - K, 0.0)
        return np.maximum(K - S, 0.0)

    # cashflows: start with maturity payoff
    cashflow = payoff(paths[:, -1])
    exercise_time = np.full(n_paths, n_steps, dtype=int)  # time index when exercised

    # work backwards: t = n_steps-1 ... 1 (skip t=0)
    for t in range(n_steps - 1, 0, -1):
        St = paths[:, t]
        immediate = payoff(St)

        # only consider paths where option is in the money at time t
        itm = immediate > 0
        if not np.any(itm):
            cashflow = cashflow * disc
            continue

        # discount existing cashflow one step to time t
        cashflow = cashflow * disc

        # regression: continuation value ~ basis(St) using ITM paths
        X = _basis_poly(St[itm], degree)
        Y = cashflow[itm]

        # least squares fit
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        continuation = X @ beta

        # decide exercise vs continue
        exercise_now = immediate[itm] > continuation

        # update those paths that exercise now
        idx_itm = np.where(itm)[0]
        ex_idx = idx_itm[exercise_now]

        cashflow[ex_idx] = immediate[ex_idx]
        exercise_time[ex_idx] = t

        # paths that did not exercise keep discounted continuation in cashflow already

    # discount cashflows from time 1 to time 0 (one more step)
    cashflow = cashflow * disc
    price = float(np.mean(cashflow))
    if return_cashflows:
        return price, cashflow
    return price