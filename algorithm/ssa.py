"""
Sparrow Search Algorithm (SSA) — Xue & Shen (2020)

Population roles:
  - Producers  (20%): search for food-rich areas, guide the group
  - Scroungers (80%): follow producers to forage
  - Vigilantes (10%): detect danger and alert the group
"""

import numpy as np


class SSA:
    def __init__(self, func, pop_size=50, max_iter=200,
                 pd=0.2, sd=0.1, st=0.8):
        self.func = func
        self.dim = func.dim
        self.bounds = func.bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.pd = pd
        self.sd = sd
        self.st = st

    def optimize(self):
        lb, ub = self.bounds
        dim = self.dim

        n_producer = max(1, int(self.pop_size * self.pd))
        n_vigilante = max(1, int(self.pop_size * self.sd))

        pos = np.random.uniform(lb, ub, (self.pop_size, dim))
        fitness = self.func.evaluate(pos)

        order = np.argsort(fitness)
        pos = pos[order]
        fitness = fitness[order]

        g_best = pos[0].copy()
        g_best_fit = fitness[0]
        convergence = [g_best_fit]

        for _ in range(self.max_iter):
            # --- Producer update ---
            for i in range(n_producer):
                alpha = 0.5 + 0.5 * np.random.rand()
                r2 = np.random.rand()
                if r2 < self.st:
                    pos[i] = pos[i] * np.exp(-i / (alpha * self.max_iter))
                else:
                    pos[i] = pos[i] + np.random.randn(dim) * (ub - lb) * 0.01
                pos[i] = np.clip(pos[i], lb, ub)

            # --- Scrounger update ---
            worst = pos[-1].copy()
            for i in range(n_producer, self.pop_size):
                if i > self.pop_size / 2:
                    pos[i] = np.random.randn(dim) * np.exp((worst - pos[i]) / (i ** 2 + 1))
                else:
                    pos[i] = pos[i] + np.random.uniform(0, 1, dim) * (pos[0] - pos[i])
                pos[i] = np.clip(pos[i], lb, ub)

            # --- Vigilante update ---
            fg = fitness[0]
            for i in range(self.pop_size - n_vigilante, self.pop_size):
                if fitness[i] > fg:
                    pos[i] = pos[0] + np.random.randn(dim) * np.abs(pos[i] - pos[0])
                else:
                    k = np.random.randint(0, self.pop_size)
                    delta = np.abs(fitness[i] - fg) + 1e-50
                    pos[i] = pos[i] + (2 * np.random.rand() - 1) * (pos[k] - pos[i]) / delta
                pos[i] = np.clip(pos[i], lb, ub)

            # --- Evaluate ---
            fitness = self.func.evaluate(pos)
            order = np.argsort(fitness)
            pos = pos[order]
            fitness = fitness[order]

            if fitness[0] < g_best_fit:
                g_best = pos[0].copy()
                g_best_fit = fitness[0]

            convergence.append(g_best_fit)

        return g_best, g_best_fit, convergence
