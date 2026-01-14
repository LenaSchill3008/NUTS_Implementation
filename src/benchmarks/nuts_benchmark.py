import numpy as np
from src.sampler.nuts import NUTSSampler
from src.sampler.rwm import RandomWalkMetropolis
from src.benchmarks.models import StandardNormal, CorrelatedGaussian, Banana, CorrelatedGaussianND
from src.benchmarks.metrics import effective_sample_size, timed_run


def run_model(name, model, x0, true_mean=None, true_cov=None):
    print(f"\n=== {name} (dim={len(x0)}) ===")

    # --- NUTS ---
    nuts = NUTSSampler(model, delta=0.65)
    (nuts_samples, stats, eps), nuts_time = timed_run(
        lambda: nuts.sample(x0, n_samples=3000, n_adapt=1000)
    )

    nuts_ess = np.array([
        effective_sample_size(nuts_samples[:, d])
        for d in range(nuts_samples.shape[1])
    ])

    print("NUTS:")
    print(f"  runtime        = {nuts_time:.2f}s")
    print(f"  eps            = {eps:.3f}")
    print(f"  ESS (min/mean) = {nuts_ess.mean():.1f}")

    if true_mean is not None:
        mean_err = np.linalg.norm(nuts_samples.mean(axis=0) - true_mean)
        print(f"  mean error     = {mean_err:.3e}")

    # --- RWM ---
    rwm = RandomWalkMetropolis(model.log_prob, step_size=0.3)
    (rwm_samples, acc), rwm_time = timed_run(
        lambda: rwm.sample(x0, n_samples=4000)
    )

    rwm_ess = np.array([
        effective_sample_size(rwm_samples[:, d])
        for d in range(rwm_samples.shape[1])
    ])

    print("RWM:")
    print(f"  runtime        = {rwm_time:.2f}s")
    print(f"  accept         = {acc:.2f}")
    print(f"  ESS (min/mean) = {rwm_ess.min():.1f} / {rwm_ess.mean():.1f}")



def run():
    """
    Run the full benchmark suite.
    """
    np.random.seed(42)

    run_model(
    "Standard Normal",
    StandardNormal(),
    x0=np.array([0.0]),
    true_mean=np.array([0.0]),
    true_cov=np.array([[1.0]])
    )

    run_model(
        "Correlated Gaussian",
        CorrelatedGaussian(),
        x0=np.zeros(2),
        true_mean=np.zeros(2)
    )

    run_model(
        "Banana Distribution",
        Banana(),
        x0=np.zeros(2)
    )

    run_model(
        "Correlated Gaussian (10D)",
        CorrelatedGaussianND(dim=10, rho=0.9),
        x0=np.zeros(10),
        true_mean=np.zeros(10)
    )

