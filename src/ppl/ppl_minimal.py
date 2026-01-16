"""
This module provides a lightweight PPL interface that allows defining models
with sample() statements. It handles constraint transformations and bridges
to the existing NUTS implementation.
"""

import numpy as np
from src.core.density import LogDensity
from src.core.gradient import NumericalGradient


_CURRENT_CONTEXT = None

def sample(name, dist, obs = None):
    """Sample from a distribution or condition on observed data."""

    if _CURRENT_CONTEXT is None:
        raise RuntimeError("sample() must be called within a context")
    return _CURRENT_CONTEXT.sample(name, dist, obs)


class Transform:
    """Base transform for bijective mappings."""

    def forward(self, x): raise NotImplementedError
    def log_det_jac(self, x): raise NotImplementedError


class Identity(Transform):
    """Identity transform for unconstrained variables."""

    def forward(self, x): return x
    def log_det_jac(self, x): return 0.0


class Exp(Transform):
    """Exponential transform: y = exp(x) for positive variables."""

    def forward(self, x): return np.exp(x)
    def log_det_jac(self, x): return np.sum(x)



class Distribution:
    """Base distribution class."""

    def log_prob(self, value): raise NotImplementedError
    def transform(self): return Identity()


class Normal(Distribution):
    """Normal distribution."""

    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale
    
    def log_prob(self, value):
        z = (value - self.loc) / self.scale
        return -0.5 * np.sum(z**2) - np.sum(np.log(self.scale)) - 0.5 * np.log(2 * np.pi)


class HalfNormal(Distribution):
    """Half-normal distribution (positive only)."""

    def __init__(self, scale=1.0):
        self.scale = scale
    
    def log_prob(self, value):
        return -0.5 * np.sum((value / self.scale)**2) - np.sum(np.log(self.scale)) + np.log(2) - 0.5 * np.log(2 * np.pi)
    
    def transform(self):
        return Exp()



class Context:
    """Base context for tracking execution."""

    def __init__(self):
        self.log_prob = 0.0
    
    def __enter__(self):
        global _CURRENT_CONTEXT
        self._prev = _CURRENT_CONTEXT
        _CURRENT_CONTEXT = self
        return self
    
    def __exit__(self, *args):
        global _CURRENT_CONTEXT
        _CURRENT_CONTEXT = self._prev
    
    def sample(self, name, dist, obs):
        raise NotImplementedError


class Collector(Context):
    """Collects variable names without requiring values."""

    def __init__(self):
        super().__init__()
        self.vars = []
    
    def sample(self, name, dist, obs):
        if obs is None:
            self.vars.append(name)
        return 0.0


class Evaluator(Context):
    """Evaluates log probability in unconstrained space."""

    def __init__(self, var_to_idx, X):
        super().__init__()
        self.var_to_idx = var_to_idx
        self.X = X
        self.constrained = {}
    
    def sample(self, name, dist, obs):
        if obs is not None:
            self.log_prob += dist.log_prob(obs)
            return obs
        
        idx = self.var_to_idx[name]
        x_raw = self.X[idx]
        
        # Transform to constrained space
        transform = dist.transform()
        value = transform.forward(x_raw)
        
        # Add log probability and Jacobian
        self.log_prob += dist.log_prob(value)
        self.log_prob += transform.log_det_jac(x_raw)
        
        self.constrained[name] = value
        return value



class LogJoint(LogDensity):
    """
    Wraps a PPL model for use with NUTS.
    
    Usage:
        def model():
            mu = sample("mu", Normal(0, 10))
            sigma = sample("sigma", HalfNormal(5))
            for y_i in data:
                sample("y", Normal(mu, sigma), obs=y_i)
        
        logjoint = LogJoint(model)
        sampler = NUTSSampler(logjoint)
        samples = sampler.sample(x0, n_samples=1000, n_adapt=500)
        constrained = logjoint.to_constrained(samples)
    """
    
    def __init__(self, model_fn, *args, **kwargs):
        self.model_fn = model_fn
        self.args = args
        self.kwargs = kwargs
        
        # Discover latent variables
        collector = Collector()
        with collector:
            try:
                model_fn(*args, **kwargs)
            except:
                pass
        
        self.vars = collector.vars
        self.var_to_idx = {v: i for i, v in enumerate(self.vars)}
        self.dim = len(self.vars)
        
        # Setup numerical gradients
        self._grad = NumericalGradient(self.log_prob, dx=1e-5, order=2)
    
    def log_prob(self, X: np.ndarray):
        """Evaluate log probability."""

        ctx = Evaluator(self.var_to_idx, X)
        with ctx:
            self.model_fn(*self.args, **self.kwargs)
        return ctx.log_prob
    
    def grad_log_prob(self, X):
        """Compute gradient via finite differences."""

        return self._grad(X)
    
    def to_constrained(self, samples):
        """
        Convert unconstrained samples to constrained space.
        """

        n = samples.shape[0]
        result = {v: [] for v in self.vars}
        
        for i in range(n):
            ctx = Evaluator(self.var_to_idx, samples[i])
            with ctx:
                self.model_fn(*self.args, **self.kwargs)
            for v, val in ctx.constrained.items():
                result[v].append(val)
        
        return {v: np.array(vals) for v, vals in result.items()}