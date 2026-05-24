"""
Particle Swarm Optimization (PSO) with linear inertia decay and velocity clamping.
"""

import numpy as np


class PSO:
    def __init__(self, func, pop_size=50, max_iter=200,
                 w_start=0.9, w_end=0.4, c1=2.0, c2=2.0, v_clamp=0.2):
        self.func = func
        self.dim = func.dim
        self.bounds = func.bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_clamp * (func.bounds[1] - func.bounds[0])

    def optimize(self):
        lb, ub = self.bounds
        dim = self.dim

        pos = np.random.uniform(lb, ub, (self.pop_size, dim))
        vel = np.random.uniform(-self.v_max, self.v_max, (self.pop_size, dim))
        p_best = pos.copy()
        p_best_fit = np.full(self.pop_size, np.inf)

        fitness = self.func.evaluate(pos)
        improved = fitness < p_best_fit
        p_best_fit[improved] = fitness[improved]
        p_best[improved] = pos[improved]

        g_best_idx = np.argmin(p_best_fit)
        g_best = p_best[g_best_idx].copy()
        g_best_fit = p_best_fit[g_best_idx]
        convergence = [g_best_fit]

        for t in range(self.max_iter):
            w = self.w_start - (self.w_start - self.w_end) * t / self.max_iter

            r1 = np.random.rand(self.pop_size, dim)
            r2 = np.random.rand(self.pop_size, dim)

            vel = (w * vel
                   + self.c1 * r1 * (p_best - pos)
                   + self.c2 * r2 * (g_best - pos))

            vel = np.clip(vel, -self.v_max, self.v_max)
            pos = pos + vel
            pos = np.clip(pos, lb, ub)

            fitness = self.func.evaluate(pos)

            improved = fitness < p_best_fit
            p_best_fit[improved] = fitness[improved]
            p_best[improved] = pos[improved]

            best_idx = np.argmin(p_best_fit)
            if p_best_fit[best_idx] < g_best_fit:
                g_best = p_best[best_idx].copy()
                g_best_fit = p_best_fit[best_idx]

            convergence.append(g_best_fit)

        return g_best, g_best_fit, convergence
