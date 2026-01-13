from abc import ABC, abstractmethod
import numpy as np


class LogDensity(ABC):
    """Abstract target distribution."""

    @abstractmethod
    def log_prob(self, x: np.ndarray) -> float:
        """Compute the log-probability of the target distribution at x."""
        pass

    @abstractmethod
    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the log-probability at x."""
        pass

    def evaluate(self, x: np.ndarray):
        """Compute the log-probability and potential energy gradient at x."""
        logp = self.log_prob(x)
        grad_u = -self.grad_log_prob(x)
        return logp, grad_u
