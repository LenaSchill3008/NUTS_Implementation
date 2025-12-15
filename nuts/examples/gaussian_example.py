import numpy as np
from core.density import LogDensity
from sampler.nuts import NUTSSampler


class CorrelatedGaussian(LogDensity):
    def __init__(self):
        self.A = np.array([[50.251256, -24.874372],
                           [-24.874372, 12.562814]])

    def log_prob(self, x):
        return -0.5 * x @ self.A @ x

    def grad_log_prob(self, x):
        return -self.A @ x


def run():
    target = CorrelatedGaussian()
    sampler = NUTSSampler(target, delta=0.2)

    x0 = np.random.randn(2)
    samples, logp, eps = sampler.sample(x0, 100_000, 5_000)

    print("Mean:", samples.mean(axis=0))
    print("Final epsilon:", eps)
