import time
import numpy as np
import matplotlib.pyplot as plt

def convergence_study(run_once, path_counts):
    prices, ses, times = [], [], []
    for n in path_counts:
        t0 = time.perf_counter()
        price, se, lo, hi = run_once(n)
        t1 = time.perf_counter()
        prices.append(price)
        ses.append(se)
        times.append(t1 - t0)
    return np.array(prices), np.array(ses), np.array(times)

def plot_convergence(path_counts, prices, ses, ref=None):
    plt.figure()
    plt.plot(path_counts, prices)
    plt.xscale("log")
    if ref is not None:
        plt.axhline(ref)
    plt.xlabel("# paths")
    plt.ylabel("price")
    plt.show()

    plt.figure()
    plt.plot(path_counts, 1.96 * ses)
    plt.xscale("log")
    plt.xlabel("# paths")
    plt.ylabel("half-width (95% CI)")
    plt.show()
