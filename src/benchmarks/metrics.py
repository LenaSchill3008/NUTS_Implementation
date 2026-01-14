import numpy as np
import time


def autocorrelation(x, lag):
    x = x - x.mean()
    return np.correlate(x[:-lag], x[lag:])[0] / np.correlate(x, x)[0]


def effective_sample_size(samples, max_lag=100):
    n = len(samples)
    rho_sum = 0.0

    for lag in range(1, max_lag):
        rho = autocorrelation(samples, lag)
        if rho <= 0:
            break
        rho_sum += rho

    tau = 1 + 2 * rho_sum
    return n / tau


def rhat(chains):
    m = len(chains)
    n = len(chains[0])
    
    chain_means = np.array([np.mean(chain) for chain in chains])
    chain_vars = np.array([np.var(chain, ddof=1) for chain in chains])
    
    B = n * np.var(chain_means, ddof=1)
    W = np.mean(chain_vars)
    
    var_plus = ((n - 1) / n) * W + (1 / n) * B
    rhat = np.sqrt(var_plus / W)
    
    return rhat


def mean_squared_jump_distance(samples):
    diffs = np.diff(samples, axis=0)
    return np.mean(np.sum(diffs ** 2, axis=1))


def compute_acceptance_rate(n_accepted, n_total):
    return n_accepted / n_total


def timed_run(fn):
    start = time.perf_counter()
    result = fn()
    runtime = time.perf_counter() - start
    return result, runtime