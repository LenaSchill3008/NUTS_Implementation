import numpy as np
from src.core.leapfrog import LeapfrogIntegrator
from src.core.tree import NutsTreeBuilder
from src.core.adaptation import DualAveraging


class NUTSSampler:
    
    def __init__(self, log_density, delta=0.6):
        self.log_density = log_density
        self.integrator = LeapfrogIntegrator(log_density)
        self.tree = NutsTreeBuilder(self.integrator)
        self.delta = delta

    def sample(self, x0, n_samples, n_adapt, collect_diagnostics=False):
        dim = len(x0)
        samples = []
        logps = []
        
        diagnostics = {
            'n_leapfrog': [],
            'depth': [],
            'accept_prob': []
        } if collect_diagnostics else None

        logp, grad_u = self.log_density.evaluate(x0)
        eps = 1.0

        adapt = DualAveraging(self.delta, eps)
        x = x0.copy()

        for m in range(1, n_samples + n_adapt + 1):
            p0 = np.random.randn(dim)
            joint0 = logp - 0.5 * p0 @ p0
            logu = joint0 - np.random.exponential()

            xm = xp = x.copy()
            pm = pp = p0.copy()
            gm = gp = grad_u.copy()

            x_new = x
            logp_new = logp

            depth = 0
            n = 1
            s = 1
            alpha = 0
            n_alpha = 0

            while s == 1:
                v = 1 if np.random.rand() < 0.5 else -1

                (
                    xm, pm, gm,
                    xp, pp, gp,
                    xc, gc, logpc,
                    nc, sc, ac, nac
                ) = self.tree.build(
                    xm if v == -1 else xp,
                    pm if v == -1 else pp,
                    gm if v == -1 else gp,
                    logu, v, depth, eps, joint0
                )

                if sc == 1 and np.random.rand() < nc / max(n + nc, 1):
                    x_new, grad_u, logp_new = xc, gc, logpc

                n += nc
                s = sc and self.tree.stop_criterion(xm, xp, pm, pp)
                alpha += ac
                n_alpha += nac
                depth += 1

            x, logp = x_new, logp_new

            if m <= n_adapt:
                eps = adapt.update(m, eps, alpha, n_alpha)
            else:
                samples.append(x.copy())
                logps.append(logp)
                
                if collect_diagnostics:
                    diagnostics['n_leapfrog'].append(nac)
                    diagnostics['depth'].append(depth)
                    diagnostics['accept_prob'].append(alpha / max(n_alpha, 1))

        result = (np.array(samples), np.array(logps), eps)
        if collect_diagnostics:
            result = result + (diagnostics,)
        
        return result