import numpy as np
from numpy import log, exp, sqrt


def find_reasonable_epsilon(log_density, x0):
    """
    Heuristic for finding an initial step size epsilon from Algorithm 4.
    
    Repeatedly doubles or halves epsilon until the acceptance 
    probability of a Langevin proposal crosses 0.5.
    """
    eps = 1.0
    r = np.random.randn(len(x0))
    
    # Compute initial log probability and gradient
    logp, grad_u = log_density.evaluate(x0)
    
    # Take one leapfrog step (using correct gradient signs)
    r_half = r - 0.5 * eps * grad_u
    x_new = x0 + eps * r_half
    logp_new, grad_u_new = log_density.evaluate(x_new)
    r_new = r_half - 0.5 * eps * grad_u_new
    
    # Compute joint probabilities (Hamiltonian)
    joint = logp - 0.5 * np.dot(r, r)
    joint_new = logp_new - 0.5 * np.dot(r_new, r_new)
    
    # Determine direction: a = 1 if we should increase eps, -1 if decrease
    # If acceptance prob > 0.5, we can use a larger eps
    log_accept_prob = joint_new - joint
    
    if log_accept_prob > np.log(0.5):
        a = 1
    else:
        a = -1
    
    # Keep doubling/halving until acceptance probability crosses 0.5
    # Condition from Algorithm 4: (p(θ',r')/p(θ,r))^a > 2^(-a)
    # Taking logs: a * log(accept_prob) > -a * log(2)
    # Rearranged: a * (log(accept_prob) + log(2)) > 0
    while a * (log_accept_prob + np.log(2)) > 0:
        eps = eps * (2**a)
        
        # Take new leapfrog step with updated eps
        r_half = r - 0.5 * eps * grad_u
        x_new = x0 + eps * r_half
        logp_new, grad_u_new = log_density.evaluate(x_new)
        r_new = r_half - 0.5 * eps * grad_u_new
        
        # Recompute joint probability
        joint_new = logp_new - 0.5 * np.dot(r_new, r_new)
        log_accept_prob = joint_new - joint
    
    return eps


class DualAveraging:
    """
    Dual averaging algorithm for step-size adaptation.
    """

    def __init__(self, delta, eps):
        # Target acceptance probability
        self.delta = delta

        # Reference point in log-epsilon space
        self.mu = log(10 * eps)

        # Hyperparameters (from the paper)
        self.gamma = 0.05
        self.t0 = 10
        self.kappa = 0.75

        # Initialize eps_bar to eps
        self.eps_bar = eps
        self.h_bar = 0.0


    def update(self, m, eps, alpha, n_alpha):
        """
        Update step size during warmup.
        """

        # Learning rate 
        eta = 1 / (m + self.t0)

        # Update running acceptance error 
        self.h_bar = (1 - eta) * self.h_bar + eta * (
            self.delta - alpha / n_alpha
        )

        # Update epsilon 
        eps = exp(self.mu - sqrt(m) / self.gamma * self.h_bar)

        # Smooth epsilon
        eta_bar = m ** (-self.kappa)
        self.eps_bar = exp(
            (1 - eta_bar) * log(self.eps_bar) + eta_bar * log(eps)
        )

        return eps