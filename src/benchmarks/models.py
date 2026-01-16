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
    
class GaussianMixture(LogDensity):
    def __init__(self):
        self.means = [np.array([-3.0, -3.0]), np.array([3.0, 3.0])]
        self.sigma = 1.0
    
    @staticmethod
    def log_sum_exp(a, b):
        max_val = max(a, b)
        return max_val + np.log(np.exp(a - max_val) + np.exp(b - max_val))
    
    def log_prob(self, x):
        logp1 = -0.5 * np.sum((x - self.means[0]) ** 2) / self.sigma ** 2
        logp2 = -0.5 * np.sum((x - self.means[1]) ** 2) / self.sigma ** 2
        return self.log_sum_exp(logp1, logp2) + np.log(0.5)
    
    def grad_log_prob(self, x):
        diff1 = x - self.means[0]
        diff2 = x - self.means[1]
        
        logp1 = -0.5 * np.sum(diff1 ** 2) / self.sigma ** 2
        logp2 = -0.5 * np.sum(diff2 ** 2) / self.sigma ** 2
        
        max_logp = max(logp1, logp2)
        w1 = np.exp(logp1 - max_logp)
        w2 = np.exp(logp2 - max_logp)
        Z = w1 + w2
        
        grad1 = -diff1 / self.sigma ** 2
        grad2 = -diff2 / self.sigma ** 2
        
        return (w1 * grad1 + w2 * grad2) / Z


class Banana(LogDensity):
    def __init__(self, b=0.1):
        self.b = b

    def log_prob(self, x):
        y1 = x[0]
        y2 = x[1] - self.b * (x[0]**2 - 1)
        return -0.5 * (y1**2 + y2**2)

    def grad_log_prob(self, x):
        y2 = x[1] - self.b * (x[0]**2 - 1)
        grad = np.zeros_like(x)
        grad[0] = -x[0] + 2 * self.b * x[0] * y2
        grad[1] = -y2
        return grad

class HighDimensionalGaussian(LogDensity):
    def __init__(self, dim=20):
        self.dim = dim
        self.sigma = np.linspace(0.1, 2.0, dim)
    
    def log_prob(self, x):
        return -0.5 * np.sum((x / self.sigma) ** 2)
    
    def grad_log_prob(self, x):
        return -x / (self.sigma ** 2)


class LogisticRegression(LogDensity):
    def __init__(self, X, y, prior_scale=10.0):
        self.X = X
        self.y = y
        self.prior_scale = prior_scale
        self.n, self.d = X.shape
    
    @staticmethod
    def log1p_exp(x):
        return np.where(x > 35, x, np.where(x < -35, 0.0, np.log1p(np.exp(np.clip(x, -35, 35)))))
    
    @staticmethod
    def sigmoid(x):
        x_clipped = np.clip(x, -35, 35)
        return np.where(x_clipped >= 0, 
                       1 / (1 + np.exp(-x_clipped)),
                       np.exp(x_clipped) / (1 + np.exp(x_clipped)))
    
    def log_prob(self, beta):
        logits = self.X @ beta
        log_likelihood = np.sum(self.y * logits - self.log1p_exp(logits))
        log_prior = -0.5 * np.sum(beta ** 2) / self.prior_scale ** 2
        return log_likelihood + log_prior
    
    def grad_log_prob(self, beta):
        logits = self.X @ beta
        probs = self.sigmoid(logits)
        grad_likelihood = self.X.T @ (self.y - probs)
        grad_prior = -beta / self.prior_scale ** 2
        return grad_likelihood + grad_prior
    

class NealsFunnel(LogDensity):
    """
    Neal's funnel distribution 
    
    The distribution is:
        y ~ Normal(0, 3)
        x_i ~ Normal(0, exp(y/2)) for i=1,...,d-1
    
    This creates a funnel shape where the scale of x depends on y:
    - Large y -> wide funnel (large variance in x)
    - Small y -> narrow neck (tiny variance in x)
    """

    def __init__(self, dim=10):
        self.dim = dim
    
    def log_prob(self, theta):
        y = theta[0]
        x = theta[1:]
        
        # Log probability of y
        log_p_y = -0.5 * (y / 3.0) ** 2
        
        # Log probability of x given y
        sigma_x = np.exp(y / 2.0)
        log_p_x = -0.5 * np.sum((x / sigma_x) ** 2) - (self.dim - 1) * y / 2.0
        
        return log_p_y + log_p_x
    
    def grad_log_prob(self, theta):
        y = theta[0]
        x = theta[1:]
        
        # Gradient w.r.t. y
        grad_y = -y / 9.0  # from p(y)
        sigma_x = np.exp(y / 2.0)
        grad_y += np.sum(x ** 2) / (2 * sigma_x ** 2) - (self.dim - 1) / 2.0  # from p(x|y)
        
        # Gradient w.r.t. x
        grad_x = -x / (sigma_x ** 2)
        
        return np.concatenate([[grad_y], grad_x])
