import numpy as np
from sampler.nuts import NUTSSampler
from sampler.rwm import RandomWalkMetropolis
from benchmarks.models import StandardNormal, CorrelatedGaussian, Banana
from benchmarks.metrics import effective_sample_size, timed_run


def run_model(name, model, x0):
    print(f"\n=== {name} ===")

    # --- NUTS ---
    nuts = NUTSSampler(model, delta=0.65)
    (nuts_samples, _, eps), nuts_time = timed_run(
        lambda: nuts.sample(x0, n_samples=3000, n_adapt=1000)
    )
    nuts_ess = effective_sample_size(nuts_samples[:, 0])

    print(f"NUTS:")
    print(f"  runtime = {nuts_time:.2f}s")
    print(f"  ESS     = {nuts_ess:.1f}")
    print(f"  eps     = {eps:.3f}")

    # --- RWM ---
    rwm = RandomWalkMetropolis(model.log_prob, step_size=0.3)
    (rwm_samples, acc), rwm_time = timed_run(
        lambda: rwm.sample(x0, n_samples=4000)
    )
    rwm_ess = effective_sample_size(rwm_samples[:, 0])

    print(f"RWM:")
    print(f"  runtime = {rwm_time:.2f}s")
    print(f"  ESS     = {rwm_ess:.1f}")
    print(f"  accept  = {acc:.2f}")


def run():
    """
    Run the full benchmark suite.
    """
    np.random.seed(42)

    # Standard Normal
    run_model("Standard Normal (1D)", StandardNormal(), x0=np.array([0.0]))

    # Correlated Gaussian
    run_model("Correlated Gaussian (2D)", CorrelatedGaussian(), x0=np.zeros(2))

    # Banana Distribution
    run_model("Banana Distribution (2D)", Banana(), x0=np.zeros(2))
