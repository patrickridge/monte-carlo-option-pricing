import time
from mcop.simulate_paths import simulate_gbm_paths
from mcop.american_lsm import american_option_lsm
from mcop.american_lsm_cpp import american_option_lsm_cpp

def main():
    S0, K, r, q, sigma, T = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0
    n_steps = 100
    n_paths = 200_000

    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, q=q, seed=123, antithetic=True)

    t0 = time.perf_counter()
    py = american_option_lsm(paths, K, r, T, is_call=False, degree=3, q=q)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    cpp = american_option_lsm_cpp(paths, K, r, T, is_call=False, degree=3)
    t3 = time.perf_counter()

    py_s = t1 - t0
    cpp_s = t3 - t2
    print(f"Python LSM: {py:.6f} in {py_s:.3f}s")
    print(f"C++    LSM: {cpp:.6f} in {cpp_s:.3f}s")
    print(f"Speedup: {py_s / cpp_s:.2f}x")

if __name__ == "__main__":
    main()
