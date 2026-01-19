import math

def american_option_crr(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    is_call: bool,
    q: float = 0.0,
) -> float:
    """
    Cox-Ross-Rubinstein binomial tree for an American option (call/put).

    Parameters
    ----------
    q : continuous dividend yield
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    dt = T / n_steps
    if dt <= 0:
        raise ValueError("dt must be positive")

    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)

    # risk-neutral probability with dividend yield q
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError("Invalid parameters: risk-neutral probability out of [0,1]")

    def payoff(S: float) -> float:
        if is_call:
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    # terminal option values
    values = [0.0] * (n_steps + 1)
    for j in range(n_steps + 1):
        ST = S0 * (u ** j) * (d ** (n_steps - j))
        values[j] = payoff(ST)

    # backward induction with early exercise
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            Sij = S0 * (u ** j) * (d ** (i - j))
            cont = disc * (p * values[j + 1] + (1.0 - p) * values[j])
            exer = payoff(Sij)
            values[j] = max(exer, cont)

    return values[0]
