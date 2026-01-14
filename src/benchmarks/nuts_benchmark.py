import numpy as np
from scipy import stats
from src.sampler.nuts import NUTSSampler
from src.sampler.rwm import RandomWalkMetropolis
from src.benchmarks.models import (
    StandardNormal, CorrelatedGaussian, Banana,
    HighDimensionalGaussian, LogisticRegression, GaussianMixture
)
from src.benchmarks.metrics import (
    effective_sample_size, rhat, mean_squared_jump_distance, timed_run
)


def generate_dispersed_starts(dim, n_chains, scale=2.0):
    return [scale * np.random.randn(dim) for _ in range(n_chains)]


def run_multiple_chains(sampler_class, model, x0_list, sampler_kwargs, sample_kwargs):
    chains = []
    times = []
    
    for x0 in x0_list:
        sampler = sampler_class(model, **sampler_kwargs)
        result, runtime = timed_run(lambda: sampler.sample(x0, **sample_kwargs))
        chains.append(result[0])
        times.append(runtime)
    
    return chains, times


def statistical_comparison(nuts_ess, rwm_ess, nuts_times, rwm_times):
    ess_ttest = stats.ttest_ind(nuts_ess, rwm_ess, alternative='greater')
    
    nuts_ess_per_sec = np.array(nuts_ess) / np.array(nuts_times)
    rwm_ess_per_sec = np.array(rwm_ess) / np.array(rwm_times)
    efficiency_ttest = stats.ttest_ind(nuts_ess_per_sec, rwm_ess_per_sec, alternative='greater')
    
    return {
        'ess_pvalue': ess_ttest.pvalue,
        'ess_statistic': ess_ttest.statistic,
        'efficiency_pvalue': efficiency_ttest.pvalue,
        'efficiency_statistic': efficiency_ttest.statistic
    }


def run_model(name, model, dim, n_chains=4):
    print(f"\n{'='*60}")
    print(f"{name}")
    print('='*60)

    x0_list = generate_dispersed_starts(dim, n_chains)

    nuts_results = []
    nuts_times = []
    for x0 in x0_list:
        sampler = NUTSSampler(model, delta=0.65)
        result, runtime = timed_run(
            lambda: sampler.sample(x0, n_samples=3000, n_adapt=1000, collect_diagnostics=True)
        )
        nuts_results.append(result)
        nuts_times.append(runtime)
    
    nuts_samples = [result[0] for result in nuts_results]
    nuts_diagnostics = [result[3] for result in nuts_results]
    
    rwm_results = []
    rwm_times = []
    for x0 in x0_list:
        sampler = RandomWalkMetropolis(model.log_prob, step_size=0.3)
        result, runtime = timed_run(lambda: sampler.sample(x0, n_samples=4000))
        rwm_results.append(result)
        rwm_times.append(runtime)
    
    rwm_samples = [result[0] for result in rwm_results]
    rwm_accept_rates = [result[1] for result in rwm_results]

    if dim == 1:
        nuts_chains_0 = [chain.flatten() for chain in nuts_samples]
        rwm_chains_0 = [chain.flatten() for chain in rwm_samples]
    else:
        nuts_chains_0 = [chain[:, 0] for chain in nuts_samples]
        rwm_chains_0 = [chain[:, 0] for chain in rwm_samples]
    
    nuts_ess = [effective_sample_size(chain) for chain in nuts_chains_0]
    rwm_ess = [effective_sample_size(chain) for chain in rwm_chains_0]
    
    nuts_rhat = rhat(nuts_chains_0)
    rwm_rhat = rhat(rwm_chains_0)
    
    nuts_msjd = np.mean([mean_squared_jump_distance(chain) for chain in nuts_samples])
    rwm_msjd = np.mean([mean_squared_jump_distance(chain) for chain in rwm_samples])
    
    avg_leapfrog = np.mean([np.mean(diag['n_leapfrog']) for diag in nuts_diagnostics])
    avg_depth = np.mean([np.mean(diag['depth']) for diag in nuts_diagnostics])
    avg_accept_prob = np.mean([np.mean(diag['accept_prob']) for diag in nuts_diagnostics])

    stats_comparison = statistical_comparison(nuts_ess, rwm_ess, nuts_times, rwm_times)

    print(f"\nNUTS (averaged over {n_chains} chains):")
    print(f"  Runtime:         {np.mean(nuts_times):.2f}s ± {np.std(nuts_times):.2f}s")
    print(f"  ESS:             {np.mean(nuts_ess):.1f} ± {np.std(nuts_ess):.1f}")
    print(f"  R-hat:           {nuts_rhat:.4f}")
    print(f"  MSJD:            {nuts_msjd:.4f}")
    print(f"  Avg leapfrogs:   {avg_leapfrog:.1f}")
    print(f"  Avg depth:       {avg_depth:.1f}")
    print(f"  Accept prob:     {avg_accept_prob:.3f}")

    print(f"\nRWM (averaged over {n_chains} chains):")
    print(f"  Runtime:         {np.mean(rwm_times):.2f}s ± {np.std(rwm_times):.2f}s")
    print(f"  ESS:             {np.mean(rwm_ess):.1f} ± {np.std(rwm_ess):.1f}")
    print(f"  R-hat:           {rwm_rhat:.4f}")
    print(f"  MSJD:            {rwm_msjd:.4f}")
    print(f"  Accept rate:     {np.mean(rwm_accept_rates):.3f} ± {np.std(rwm_accept_rates):.3f}")

    print(f"\nStatistical Comparison:")
    print(f"  ESS difference:  t={stats_comparison['ess_statistic']:.2f}, p={stats_comparison['ess_pvalue']:.4f}")
    print(f"  Efficiency:      t={stats_comparison['efficiency_statistic']:.2f}, p={stats_comparison['efficiency_pvalue']:.4f}")
    
    if stats_comparison['ess_pvalue'] < 0.05:
        print(f"  → NUTS has significantly higher ESS (p < 0.05)")
    if stats_comparison['efficiency_pvalue'] < 0.05:
        print(f"  → NUTS is significantly more efficient (p < 0.05)")


def run():
    np.random.seed(42)

    run_model("Standard Normal (1D)", StandardNormal(), dim=1)
    run_model("Correlated Gaussian (2D)", CorrelatedGaussian(), dim=2)
    run_model("Banana Distribution (2D)", Banana(), dim=2)
    run_model("High-Dimensional Gaussian (20D)", HighDimensionalGaussian(20), dim=20)
    run_model("Gaussian Mixture (2D)", GaussianMixture(), dim=2)
    

    # print("Logistic Regression")
    
    np.random.seed(123)
    n, d = 200, 5
    X = np.random.randn(n, d)
    true_beta = np.random.randn(d)
    logits = X @ true_beta
    y = (np.random.rand(n) < 1 / (1 + np.exp(-logits))).astype(float)
    
    model = LogisticRegression(X, y)
    run_model("Logistic Regression (5D)", model, dim=d, n_chains=4)
    
    print(f"\n{'='*60}")
    print("Benchmark Complete")
    print('='*60)