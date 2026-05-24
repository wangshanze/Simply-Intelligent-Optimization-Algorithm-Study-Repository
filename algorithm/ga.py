"""
Genetic Algorithm (GA) with roulette-wheel selection, SBX crossover, and polynomial mutation.
"""

import numpy as np


class GA:
    def __init__(self, func, pop_size=50, max_iter=200,
                 pc=0.9, pm=None, eta_c=15, eta_m=20, elite_rate=0.04):
        self.func = func
        self.dim = func.dim
        self.bounds = func.bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.pc = pc
        self.pm = pm if pm is not None else 1.0 / func.dim
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.elite_count = max(1, int(pop_size * elite_rate))

    def _clamp(self, x):
        return np.clip(x, self.bounds[0], self.bounds[1])

    def _roulette_select(self, pop, fitness):
        inv_fit = 1.0 / (fitness - fitness.min() + 1e-50)
        prob = inv_fit / inv_fit.sum()
        idx = np.random.choice(len(pop), size=len(pop), p=prob)
        return pop[idx].copy()

    def _sbx_crossover(self, p1, p2):
        u = np.random.rand(self.dim)
        beta = np.where(
            u <= 0.5,
            (2 * u) ** (1 / (self.eta_c + 1)),
            (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1))
        )
        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        return self._clamp(c1), self._clamp(c2)

    def _polynomial_mutation(self, x):
        u = np.random.rand(self.dim)
        delta = np.where(
            u < 0.5,
            (2 * u) ** (1 / (self.eta_m + 1)) - 1,
            1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1))
        )
        lb, ub = self.bounds
        return self._clamp(x + delta * (ub - lb))

    def optimize(self):
        lb, ub = self.bounds
        dim = self.dim

        pop = np.random.uniform(lb, ub, (self.pop_size, dim))
        fitness = self.func.evaluate(pop)
        best_idx = np.argmin(fitness)
        g_best = pop[best_idx].copy()
        g_best_fit = fitness[best_idx]
        convergence = [g_best_fit]

        for _ in range(self.max_iter):
            # 1. Roulette-wheel selection
            mating_pool = self._roulette_select(pop, fitness)

            # 2. Crossover
            offspring = []
            for i in range(0, self.pop_size, 2):
                if i + 1 >= self.pop_size:
                    offspring.append(mating_pool[i])
                    break
                if np.random.rand() < self.pc:
                    c1, c2 = self._sbx_crossover(mating_pool[i], mating_pool[i + 1])
                    offspring.extend([c1, c2])
                else:
                    offspring.extend([mating_pool[i], mating_pool[i + 1]])
            offspring = np.array(offspring[:self.pop_size])

            # 3. Mutation
            for i in range(self.pop_size):
                if np.random.rand() < self.pm:
                    offspring[i] = self._polynomial_mutation(offspring[i])

            # 4. Elitism — preserve best parents, discard same number of worst offspring
            elite_idx = np.argsort(fitness)[:self.elite_count]
            worst_off_idx = np.argsort(self.func.evaluate(offspring))[::-1][:self.elite_count]
            offspring[worst_off_idx] = pop[elite_idx]

            pop = offspring
            fitness = self.func.evaluate(pop)

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < g_best_fit:
                g_best = pop[best_idx].copy()
                g_best_fit = fitness[best_idx]

            convergence.append(g_best_fit)

        return g_best, g_best_fit, convergence
