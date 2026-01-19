from dataclasses import dataclass

@dataclass(frozen=True)
class GBMParams:
    S0: float
    r: float
    sigma: float
    T: float
    q: float = 0.0
