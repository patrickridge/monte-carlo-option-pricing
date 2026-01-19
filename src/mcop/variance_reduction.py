import numpy as np

def control_variate_adjustment(
    x: np.ndarray,
    y: np.ndarray,
    y_true_mean: float,
) -> tuple[float, float]:
    """
    Control variate estimator for E[x] using control y with known E[y].

    Returns
    -------
    x_cv_mean : float
        Control variate-adjusted estimate of E[x]
    beta_hat : float
        Estimated optimal beta
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    x_mean = x.mean()
    y_mean = y.mean()

    var_y = np.mean((y - y_mean) ** 2)
    if var_y == 0.0:
        return float(x_mean), 0.0

    cov_xy = np.mean((x - x_mean) * (y - y_mean))
    beta = cov_xy / var_y

    x_cv = x - beta * (y - y_true_mean)
    return float(x_cv.mean()), float(beta)


def mc_mean_se(x: np.ndarray) -> tuple[float, float]:
    """
    Return mean and standard error.
    """
    x = np.asarray(x, dtype=float)
    mean = float(x.mean())
    se = float(x.std(ddof=1)) / np.sqrt(x.size)
    return mean, se
