import numpy as np
from src.core.density import LogDensity


class StandardNormal(LogDensity):
    def log_prob(self, x):
        return -0.5 * np.dot(x, x)

    def grad_log_prob(self, x):
        return -x


class CorrelatedGaussian(LogDensity):
    def __init__(self):
        self.A = np.array([[50.0, -24.0],
                           [-24.0, 12.0]])

    def log_prob(self, x):
        return -0.5 * x @ self.A @ x

    def grad_log_prob(self, x):
        return -self.A @ x


class Banana(LogDensity):
    def __init__(self, b=0.1):
        self.b = b

    def log_prob(self, x):
        y1 = x[0]
        y2 = x[1] - self.b * (x[0]**2 - 1)
        return -0.5 * (y1**2 + y2**2)

    def grad_log_prob(self, x):
        grad = np.zeros_like(x)
        grad[1] = -(x[1] - self.b * (x[0]**2 - 1))
        grad[0] = -x[0] + 2 * self.b * x[0] * grad[1]
        return grad
