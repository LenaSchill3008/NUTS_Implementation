import numpy as np


class NumericalGradient:
    """Compute the numerical gradient of a scalar function using finite differences."""
    def __init__(self, f, dx=1e-3, order=2):
        self.f = f
        self.dx = dx
        self.order = order

    def __call__(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(x)
        for i in range(len(x)):
            d = np.zeros_like(x)
            d[i] = self.dx

            if self.order == 1:
                grad[i] = (self.f(x + d) - self.f(x - d)) / (2 * self.dx)
            else:
                grad[i] = (
                    self.f(x - 2*d)
                    - 8*self.f(x - d)
                    + 8*self.f(x + d)
                    - self.f(x + 2*d)
                ) / (12 * self.dx)

        return grad
