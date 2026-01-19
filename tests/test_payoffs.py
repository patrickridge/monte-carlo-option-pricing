import numpy as np
from mcop.payoffs import european_call, european_put

def test_european_call_put_payoffs():
    paths = np.array([
        [100.0, 90.0],
        [100.0, 100.0],
        [100.0, 110.0],
    ])
    K = 100.0
    call = european_call(paths, K)
    put = european_put(paths, K)

    assert np.allclose(call, [0.0, 0.0, 10.0])
    assert np.allclose(put, [10.0, 0.0, 0.0])
