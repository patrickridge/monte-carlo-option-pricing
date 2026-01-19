import numpy as np

def control_variate_adjustment(
    x: np.ndarray,
    y: np.ndarray,
    y_true_mean: float,
) -> tuple[float, float]:
    """
    Control variate estimator for E[x] using control y with known E[y]=y_true_mean.

    Returns
    -------
    x_cv_mean : float
        Control variate adjusted estimate of E[x].
    beta_hat : float
        Estimated optimal beta = Cov(x,y)/Var(y).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    y_centered = y - y.mean()
    var_y = float(np.mean(y_centered ** 2))
    if var_y == 0.0:
        return float(x.mean()), 0.0

    cov_xy = float(np.mean((x - x.mean()) * (y - y.mean())))
    beta = cov_xy / var_y

    x_cv = x - beta * (y - y_true_mean)
    return float(x_cv.mean()), float(beta)

def mc_mean_se_ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float, float]:
    """
    Mean + SE + (approx) normal CI for sample x.
    """
    x = np.asarray(x, dtype=float)
    mean = float(x.mean())
    s = float(x.std(ddof=1))
    se = s / np.sqrt(x.size)
    z = 1.959963984540054  # 97.5% quantile
    return mean, se, mean - z * se, mean + z * se
