"""
Minimal PPL Demo

Shows how to use the minimal PPL with NUTS for Bayesian inference.
"""

import numpy as np
from src.ppl_minimal import sample, Normal, HalfNormal, LogJoint
from src.sampler.nuts import NUTSSampler


def demo_simple():
    """Demo 1: Infer mean and variance from data."""
    print("\n" + "="*60)
    print("Demo 1: Simple Inference")
    print("="*60)
    
    # Generate data
    np.random.seed(42)
    true_mu, true_sigma = 2.5, 1.2
    data = np.random.normal(true_mu, true_sigma, 100)
    
    # Define model
    def model():
        mu = sample("mu", Normal(0, 10))
        sigma = sample("sigma", HalfNormal(10))
        for y_i in data:
            sample("y", Normal(mu, sigma), obs=y_i)
    
    # Run inference
    logjoint = LogJoint(model)
    print(f"Model: {logjoint.dim}D, variables: {logjoint.vars}")
    
    sampler = NUTSSampler(logjoint, delta=0.65)
    samples, _, _, diag = sampler.sample(
        np.zeros(logjoint.dim), n_samples=1000, n_adapt=500, collect_diagnostics=True
    )
    
    # Get constrained samples
    result = logjoint.to_constrained(samples)
    
    print(f"\nTrue mu: {true_mu:.2f}, Estimated: {np.mean(result['mu']):.2f} ± {np.std(result['mu']):.2f}")
    print(f"True sigma: {true_sigma:.2f}, Estimated: {np.mean(result['sigma']):.2f} ± {np.std(result['sigma']):.2f}")
    print(f"Acceptance rate: {np.mean(diag['accept_prob']):.3f}")


def demo_regression():
    """Demo 2: Bayesian linear regression."""
    print("\n" + "="*60)
    print("Demo 2: Bayesian Linear Regression")
    print("="*60)
    
    # Generate data
    np.random.seed(123)
    n, d = 50, 3
    X = np.random.randn(n, d)
    true_beta = np.array([1.5, -0.8, 0.5])
    true_sigma = 0.5
    y = X @ true_beta + np.random.normal(0, true_sigma, n)
    
    # Define model
    def model():
        beta = [sample(f"beta_{j}", Normal(0, 10)) for j in range(d)]
        sigma = sample("sigma", HalfNormal(5))
        y_pred = X @ np.array(beta)
        for i in range(n):
            sample(f"y_{i}", Normal(y_pred[i], sigma), obs=y[i])
    
    # Run inference
    logjoint = LogJoint(model)
    sampler = NUTSSampler(logjoint, delta=0.65)
    samples, _, _ = sampler.sample(
        np.zeros(logjoint.dim), n_samples=1000, n_adapt=500, collect_diagnostics=False
    )
    
    # Get results
    result = logjoint.to_constrained(samples)
    
    print(f"\n{'Parameter':<12} {'True':<8} {'Mean':<8} {'Std':<8}")
    print("-" * 40)
    for j in range(d):
        print(f"beta_{j:<6} {true_beta[j]:>7.2f} {np.mean(result[f'beta_{j}']):>7.2f} {np.std(result[f'beta_{j}']):>7.2f}")
    print(f"{'sigma':<12} {true_sigma:>7.2f} {np.mean(result['sigma']):>7.2f} {np.std(result['sigma']):>7.2f}")


if __name__ == "__main__":
    demo_simple()
    demo_regression()
    
    print("\n" + "="*60)
    print("All demos complete!")
    print("="*60)