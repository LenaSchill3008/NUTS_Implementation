import numpy as np


class RandomWalkMetropolis:
    """
    Random-Walk Metropolis baseline sampler.
    """

    def __init__(self, log_prob, step_size):
        self.log_prob = log_prob
        self.step_size = step_size

    def sample(self, x0, n_samples):
        x = x0.copy()
        logp = self.log_prob(x)

        samples = []
        accepts = 0

        for _ in range(n_samples):
            # Propose local Gaussian move
            proposal = x + self.step_size * np.random.randn(*x.shape)
            logp_prop = self.log_prob(proposal)

            # Metropolis acceptance step
            if np.log(np.random.rand()) < logp_prop - logp:
                x = proposal
                logp = logp_prop
                accepts += 1

            samples.append(x.copy())

        acceptance_rate = accepts / n_samples
        return np.array(samples), acceptance_rate
