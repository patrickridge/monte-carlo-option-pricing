import numpy as np
from mcop.variance_reduction import control_variate_adjustment, mc_mean_se

def test_control_variate_reduces_variance():
    rng = np.random.default_rng(0)
    n = 50_000

    # Control with known mean 0
    y = rng.normal(size=n)
    x = 2.5 * y + rng.normal(scale=1.0, size=n)

    mean_plain, se_plain = mc_mean_se(x)

    mean_cv, beta = control_variate_adjustment(x, y, y_true_mean=0.0)
    x_cv = x - beta * (y - 0.0)
    mean_cv2, se_cv = mc_mean_se(x_cv)

    assert se_cv < 0.7 * se_plain
    assert abs(mean_cv - mean_plain) < 0.05
