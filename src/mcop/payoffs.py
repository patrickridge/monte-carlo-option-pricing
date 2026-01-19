import numpy as np

def european_call(paths: np.ndarray, K: float) -> np.ndarray:
    ST = paths[:, -1]
    return np.maximum(ST - K, 0.0)

def european_put(paths: np.ndarray, K: float) -> np.ndarray:
    ST = paths[:, -1]
    return np.maximum(K - ST, 0.0)
