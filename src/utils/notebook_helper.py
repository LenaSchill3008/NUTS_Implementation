import numpy as np
import matplotlib.pyplot as plt

from src.sampler.nuts import NUTSSampler
from src.sampler.hmc import HMCSampler
from src.sampler.rwm import RandomWalkMetropolis

from src.benchmarks.metrics import (
    effective_sample_size, rhat, mean_squared_jump_distance, 
    autocorrelation, timed_run
)

def run_sampler_comparison(model, dim, n_samples=3000, n_adapt=1000, n_chains=4):
    x0_list = [2.0 * np.random.randn(dim) for _ in range(n_chains)]
    
    nuts_samples = []
    nuts_times = []
    nuts_diagnostics = []
    
    for x0 in x0_list:
        sampler = NUTSSampler(model, delta=0.65)
        result, runtime = timed_run(
            lambda: sampler.sample(x0, n_samples=n_samples, n_adapt=n_adapt, collect_diagnostics=True)
        )
        nuts_samples.append(result[0])
        nuts_times.append(runtime)
        nuts_diagnostics.append(result[3])
    
    hmc_samples = []
    hmc_times = []
    hmc_accept_rates = []
    
    for x0 in x0_list:
        sampler = HMCSampler(model, L=10, eps=0.1)
        result, runtime = timed_run(lambda: sampler.sample(x0, n_samples=n_samples + n_adapt))
        hmc_samples.append(result[0])
        hmc_times.append(runtime)
        hmc_accept_rates.append(result[1])
    
    rwm_samples = []
    rwm_times = []
    rwm_accept_rates = []
    
    for x0 in x0_list:
        sampler = RandomWalkMetropolis(model.log_prob, step_size=0.3)
        result, runtime = timed_run(lambda: sampler.sample(x0, n_samples=n_samples + n_adapt))
        rwm_samples.append(result[0])
        rwm_times.append(runtime)
        rwm_accept_rates.append(result[1])
    
    return {
        'nuts_samples': nuts_samples,
        'nuts_times': nuts_times,
        'nuts_diagnostics': nuts_diagnostics,
        'hmc_samples': hmc_samples,
        'hmc_times': hmc_times,
        'hmc_accept_rates': hmc_accept_rates,
        'rwm_samples': rwm_samples,
        'rwm_times': rwm_times,
        'rwm_accept_rates': rwm_accept_rates
    }

def plot_traces(nuts_samples, hmc_samples, rwm_samples, title, dim_idx=0):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    for chain in nuts_samples:
        if chain.ndim == 1:
            axes[0].plot(chain, alpha=0.7, linewidth=0.5)
        else:
            axes[0].plot(chain[:, dim_idx], alpha=0.7, linewidth=0.5)
    axes[0].set_title(f'NUTS - {title}')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel(f'x[{dim_idx}]')
    axes[0].grid(True, alpha=0.3)
    
    for chain in hmc_samples:
        if chain.ndim == 1:
            axes[1].plot(chain, alpha=0.7, linewidth=0.5)
        else:
            axes[1].plot(chain[:, dim_idx], alpha=0.7, linewidth=0.5)
    axes[1].set_title(f'HMC - {title}')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel(f'x[{dim_idx}]')
    axes[1].grid(True, alpha=0.3)
    
    for chain in rwm_samples:
        if chain.ndim == 1:
            axes[2].plot(chain, alpha=0.7, linewidth=0.5)
        else:
            axes[2].plot(chain[:, dim_idx], alpha=0.7, linewidth=0.5)
    axes[2].set_title(f'RWM - {title}')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel(f'x[{dim_idx}]')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_autocorrelation(nuts_samples, hmc_samples, rwm_samples, title, max_lag=100, dim_idx=0):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    for chain in nuts_samples:
        if chain.ndim == 1:
            chain_data = chain
        else:
            chain_data = chain[:, dim_idx]
        acf = [autocorrelation(chain_data, lag) for lag in range(1, min(max_lag, len(chain_data)))]
        axes[0].plot(range(1, len(acf) + 1), acf, alpha=0.7)
    axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[0].set_title(f'NUTS Autocorrelation - {title}')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Autocorrelation')
    axes[0].grid(True, alpha=0.3)
    
    for chain in hmc_samples:
        if chain.ndim == 1:
            chain_data = chain
        else:
            chain_data = chain[:, dim_idx]
        acf = [autocorrelation(chain_data, lag) for lag in range(1, min(max_lag, len(chain_data)))]
        axes[1].plot(range(1, len(acf) + 1), acf, alpha=0.7)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1].set_title(f'HMC Autocorrelation - {title}')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Autocorrelation')
    axes[1].grid(True, alpha=0.3)
    
    for chain in rwm_samples:
        if chain.ndim == 1:
            chain_data = chain
        else:
            chain_data = chain[:, dim_idx]
        acf = [autocorrelation(chain_data, lag) for lag in range(1, min(max_lag, len(chain_data)))]
        axes[2].plot(range(1, len(acf) + 1), acf, alpha=0.7)
    axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[2].set_title(f'RWM Autocorrelation - {title}')
    axes[2].set_xlabel('Lag')
    axes[2].set_ylabel('Autocorrelation')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_2d_samples(nuts_samples, hmc_samples, rwm_samples, title, true_samples=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    nuts_combined = np.vstack(nuts_samples)
    axes[0].scatter(nuts_combined[:, 0], nuts_combined[:, 1], alpha=0.3, s=1)
    if true_samples is not None:
        axes[0].scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.1, s=1, c='red', label='True')
    axes[0].set_title(f'NUTS - {title}')
    axes[0].set_xlabel('x[0]')
    axes[0].set_ylabel('x[1]')
    axes[0].grid(True, alpha=0.3)
    
    hmc_combined = np.vstack(hmc_samples)
    axes[1].scatter(hmc_combined[:, 0], hmc_combined[:, 1], alpha=0.3, s=1)
    if true_samples is not None:
        axes[1].scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.1, s=1, c='red', label='True')
    axes[1].set_title(f'HMC - {title}')
    axes[1].set_xlabel('x[0]')
    axes[1].set_ylabel('x[1]')
    axes[1].grid(True, alpha=0.3)
    
    rwm_combined = np.vstack(rwm_samples)
    axes[2].scatter(rwm_combined[:, 0], rwm_combined[:, 1], alpha=0.3, s=1)
    if true_samples is not None:
        axes[2].scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.1, s=1, c='red', label='True')
    axes[2].set_title(f'RWM - {title}')
    axes[2].set_xlabel('x[0]')
    axes[2].set_ylabel('x[1]')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compute_metrics(results, dim):
    nuts_samples = results['nuts_samples']
    hmc_samples = results['hmc_samples']
    rwm_samples = results['rwm_samples']
    
    if dim == 1:
        nuts_chains_0 = [chain.flatten() for chain in nuts_samples]
        hmc_chains_0 = [chain.flatten() for chain in hmc_samples]
        rwm_chains_0 = [chain.flatten() for chain in rwm_samples]
    else:
        nuts_chains_0 = [chain[:, 0] for chain in nuts_samples]
        hmc_chains_0 = [chain[:, 0] for chain in hmc_samples]
        rwm_chains_0 = [chain[:, 0] for chain in rwm_samples]
    
    nuts_ess = [effective_sample_size(chain) for chain in nuts_chains_0]
    hmc_ess = [effective_sample_size(chain) for chain in hmc_chains_0]
    rwm_ess = [effective_sample_size(chain) for chain in rwm_chains_0]
    
    nuts_rhat = rhat(nuts_chains_0)
    hmc_rhat = rhat(hmc_chains_0)
    rwm_rhat = rhat(rwm_chains_0)
    
    nuts_msjd = np.mean([mean_squared_jump_distance(chain) for chain in nuts_samples])
    hmc_msjd = np.mean([mean_squared_jump_distance(chain) for chain in hmc_samples])
    rwm_msjd = np.mean([mean_squared_jump_distance(chain) for chain in rwm_samples])
    
    nuts_ess_per_sec = np.array(nuts_ess) / np.array(results['nuts_times'])
    hmc_ess_per_sec = np.array(hmc_ess) / np.array(results['hmc_times'])
    rwm_ess_per_sec = np.array(rwm_ess) / np.array(results['rwm_times'])
    
    return {
        'nuts_ess': nuts_ess,
        'hmc_ess': hmc_ess,
        'rwm_ess': rwm_ess,
        'nuts_rhat': nuts_rhat,
        'hmc_rhat': hmc_rhat,
        'rwm_rhat': rwm_rhat,
        'nuts_msjd': nuts_msjd,
        'hmc_msjd': hmc_msjd,
        'rwm_msjd': rwm_msjd,
        'nuts_ess_per_sec': nuts_ess_per_sec,
        'hmc_ess_per_sec': hmc_ess_per_sec,
        'rwm_ess_per_sec': rwm_ess_per_sec,
        'nuts_time': np.mean(results['nuts_times']),
        'hmc_time': np.mean(results['hmc_times']),
        'rwm_time': np.mean(results['rwm_times'])
    }