import numpy as np


class LeapfrogIntegrator:
    """
    Implements a single leapfrog integration step.
    """
    def __init__(self, log_density):
        self.log_density = log_density # The target distribution

    def step(self, x, p, grad_u, eps):
        """ 
        Perform one leapfrog step.
        Args:
            x: Current position.
            p: Current momentum.
            grad_u: Gradient of the potential energy at position x.
            eps: Step size.
        Returns:
            x_new: New position after the leapfrog step.
            p_new: New momentum after the leapfrog step .
            grad_u_new: Gradient of the potential energy at new position.
            logp: Log density at new position.
        """
        # Half-step momentum update
        p_half = p + 0.5 * eps * grad_u

        # Full-step position update
        x_new = x + eps * p_half

        # Recompute log-probability and gradient at new position
        logp, grad_u_new = self.log_density.evaluate(x_new)

        # Second half-step momentum update
        p_new = p_half + 0.5 * eps * grad_u_new
        
        return x_new, p_new, grad_u_new, logp
