import numpy as np
from mcop.variance_reduction import control_variate_adjustment, mc_mean_se_ci

def test_control_variate_reduces_variance_on_correlated_data():
    rng = np.random.default_rng(0)
    n = 50_000

    # y has known mean 0, x is highly correlated with y + noise
    y = rng.normal(size=n)
    x = 3.0 * y + rng.normal(scale=0.5, size=n)

    mean_plain, se_plain, *_ = mc_mean_se_ci(x)
    mean_cv, beta = control_variate_adjustment(x, y, y_true_mean=0.0)
    mean_cv2, se_cv, *_ = mc_mean_se_ci(x - beta * (y - 0.0))

    # same estimate (up to noise) but lower SE
    assert se_cv < 0.6 * se_plain
    assert abs(mean_cv - mean_plain) < 0.05
