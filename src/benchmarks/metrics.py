import numpy as np
import time


def autocorrelation(x, lag):
    x = x - x.mean()
    return np.correlate(x[:-lag], x[lag:])[0] / np.correlate(x, x)[0]


def effective_sample_size(samples, max_lag=100):
    """
    Very simple ESS estimate using autocorrelation time.
    """
    n = len(samples)
    rho_sum = 0.0

    for lag in range(1, max_lag):
        rho = autocorrelation(samples, lag)
        if rho <= 0:
            break
        rho_sum += rho

    tau = 1 + 2 * rho_sum
    return n / tau


def timed_run(fn):
    start = time.perf_counter()
    result = fn()
    runtime = time.perf_counter() - start
    return result, runtime
