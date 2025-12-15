import numpy as np


class NutsTreeBuilder:
    """ 
    Recursive tree builder for the No-U-Turn Sampler.
    """
    def __init__(self, integrator):
        # Leapfrog integrator used to simulate Hamiltonian dynamics
        self.integrator = integrator

    @staticmethod
    def stop_criterion(x_minus, x_plus, p_minus, p_plus):
        """No-U-Turn stopping criterion. Checks whether the trajectory has started doubling back on itself (a "U-turn")."""
        dx = x_plus - x_minus
        return (dx @ p_minus >= 0) and (dx @ p_plus >= 0)

    def build(
        self, x, p, grad_u, logu, v, depth, eps, joint0
    ):
        """Recursively builds a binary tree of leapfrog steps."""
        if depth == 0:
            x1, p1, g1, logp1 = self.integrator.step(
                x, p, grad_u, v * eps
            )
            # Compute Hamiltonian
            joint = logp1 - 0.5 * np.dot(p1, p1)

             # Whether this state is valid under the slice
            n = int(logu < joint)

            # Numerical safety check
            s = int(logu - 1000 < joint)

             # Acceptance probability for dual averaging
            alpha = min(1.0, np.exp(joint - joint0))
            return (
                x1, p1, g1, # leftmost state
                x1, p1, g1, # rightmost state
                x1, g1, logp1, # proposal
                n, s, alpha, 1
            )

        # build the left subtree
        (
            xm, pm, gm,
            xp, pp, gp,
            x1, g1, logp1,
            n, s, a, na
        ) = self.build(x, p, grad_u, logu, v, depth - 1, eps, joint0)

        if s == 1:
            # build the right subtree
            if v == -1:
                (
                    xm, pm, gm,
                    _, _, _,
                    x2, g2, logp2,
                    n2, s2, a2, na2
                ) = self.build(xm, pm, gm, logu, v, depth - 1, eps, joint0)
            else:
                (
                    _, _, _,
                    xp, pp, gp,
                    x2, g2, logp2,
                    n2, s2, a2, na2
                ) = self.build(xp, pp, gp, logu, v, depth - 1, eps, joint0)

            # choose which proposal to keep
            if np.random.rand() < n2 / max(n + n2, 1):
                x1, g1, logp1 = x2, g2, logp2

            # update counts and stopping criterion
            n += n2
            s = s2 and self.stop_criterion(xm, xp, pm, pp)
            a += a2
            na += na2

        return xm, pm, gm, xp, pp, gp, x1, g1, logp1, n, s, a, na
