import numpy as np

def mc_price(payoffs: np.ndarray, r: float, T: float) -> tuple[float, float, float, float]:
    """
    Discounted Monte Carlo estimator with standard error and 95% CI (normal approx).

    Returns (price, se, ci_low, ci_high)
    """
    disc = np.exp(-r * T)
    x = disc * payoffs

    price = float(x.mean())
    s = float(x.std(ddof=1))
    se = s / np.sqrt(x.size)

    z = 1.959963984540054  # ~N(0,1) 97.5% quantile
    ci_low = price - z * se
    ci_high = price + z * se
    return price, se, ci_low, ci_high
