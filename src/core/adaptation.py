from numpy import log, exp, sqrt


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

        # Running averages
        self.eps_bar = 1.0
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
        eta = m ** -self.kappa
        self.eps_bar = exp(
            (1 - eta) * log(self.eps_bar) + eta * log(eps)
        )

        return eps
