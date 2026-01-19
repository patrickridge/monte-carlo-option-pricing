import numpy as np
from . import _mcop_cpp

def american_option_lsm_cpp(
    paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    is_call: bool,
    degree: int = 2,
) -> float:
    """
    C++ LSM pricer using pre-simulated paths.
    """
    paths = np.ascontiguousarray(paths, dtype=float)
    return float(_mcop_cpp.lsm_price_from_paths(paths, K, r, T, is_call, degree))
