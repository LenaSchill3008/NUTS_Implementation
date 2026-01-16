import numpy as np
from src.core.leapfrog import LeapfrogIntegrator


class HMCSampler:
    """
    Hamiltonian Monte Carlo sampler.
    """

    def __init__(self, log_density, L, eps):
        self.log_density = log_density
        self.L = L
        self.eps = eps
        self.integrator = LeapfrogIntegrator(log_density)

    def sample(self, x0, n_samples):
        x = x0.copy()
        dim = len(x0)
        samples = []
        accepts = 0

        logp, grad_u = self.log_density.evaluate(x)

        for _ in range(n_samples):
            p = np.random.randn(dim)
            
            H_current = -logp + 0.5 * np.dot(p, p)
            
            x_prop = x.copy()
            p_prop = p.copy()
            grad_u_prop = grad_u.copy()
            
            for _ in range(self.L):
                x_prop, p_prop, grad_u_prop, logp_prop = self.integrator.step(
                    x_prop, p_prop, grad_u_prop, self.eps
                )
            
            p_prop = -p_prop
            
            H_proposed = -logp_prop + 0.5 * np.dot(p_prop, p_prop)
            
            if np.log(np.random.rand()) < H_current - H_proposed:
                x = x_prop
                logp = logp_prop
                grad_u = grad_u_prop
                accepts += 1

            samples.append(x.copy())

        acceptance_rate = accepts / n_samples
        return np.array(samples), acceptance_rate