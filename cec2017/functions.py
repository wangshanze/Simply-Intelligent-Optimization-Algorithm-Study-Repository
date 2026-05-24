"""
10 benchmark test functions for optimization algorithm testing.

Each function is a class with:
    - __init__(dim): sets dimension and search bounds
    - evaluate(x):   returns the function value at x, where x is a numpy array of shape (dim,) or (n, dim)
    - name:           string identifier

All functions have a global minimum of 0 (at the origin or a known point).
"""

import numpy as np


class Sphere:
    """f(x) = sum(x_i^2), search range [-100, 100]^D, global min f(0)=0."""

    def __init__(self, dim=30):
        self.dim = dim
        self.bounds = [-100, 100]
        self.name = "Sphere"

    def evaluate(self, x):
        x = np.asarray(x)
        return np.sum(x ** 2, axis=-1)


class BentCigar:
    """f(x) = x_1^2 + 10^6 * sum_{i=2}^D x_i^2, search range [-100, 100]^D."""

    def __init__(self, dim=30):
        self.dim = dim
        self.bounds = [-100, 100]
        self.name = "BentCigar"

    def evaluate(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            return x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2)
        return x[:, 0] ** 2 + 1e6 * np.sum(x[:, 1:] ** 2, axis=-1)


class Zakharov:
    """f(x) = sum(x_i^2) + (0.5 * sum(i*x_i))^2 + (0.5 * sum(i*x_i))^4, range [-5,10]^D."""

    def __init__(self, dim=30):
        self.dim = dim
        self.bounds = [-5, 10]
        self.name = "Zakharov"

    def evaluate(self, x):
        x = np.asarray(x)
        i = np.arange(1, self.dim + 1)
        sum1 = np.sum(x ** 2, axis=-1)
        sum2 = np.dot(x, i) if x.ndim == 1 else np.dot(x, i)
        return sum1 + (0.5 * sum2) ** 2 + (0.5 * sum2) ** 4


class Rosenbrock:
    """f(x) = sum_{i=1}^{D-1} [100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2], range [-100, 100]^D.
    Global minimum f(1,1,...,1) = 0.
    """

    def __init__(self, dim=30):
        self.dim = dim
        self.bounds = [-100, 100]
        self.name = "Rosenbrock"
        self._optimum = np.ones(dim)

    def evaluate(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        a = x[:, :-1]
        b = x[:, 1:]
        result = np.sum(100 * (b - a ** 2) ** 2 + (a - 1) ** 2, axis=-1)
        if result.shape == (1,):
            return result.item()
        return result


class Rastrigin:
    """f(x) = 10*D + sum(x_i^2 - 10*cos(2*pi*x_i)), range [-5.12, 5.12]^D.
    Global minimum f(0) = 0.
    """

    def __init__(self, dim=30):
        self.dim = dim
        self.bounds = [-5.12, 5.12]
        self.name = "Rastrigin"

    def evaluate(self, x):
        x = np.asarray(x)
        return 10 * self.dim + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=-1)


class Ackley:
    """Ackley function, range [-32, 32]^D, global min f(0)=0."""

    def __init__(self, dim=30):
        self.dim = dim
        self.bounds = [-32, 32]
        self.name = "Ackley"

    def evaluate(self, x):
        x = np.asarray(x)
        d = self.dim
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2, axis=-1) / d))
        term2 = -np.exp(np.sum(np.cos(2 * np.pi * x), axis=-1) / d)
        return term1 + term2 + 20 + np.e


class Griewank:
    """f(x) = 1 + sum(x_i^2)/4000 - prod(cos(x_i / sqrt(i))), range [-600, 600]^D."""

    def __init__(self, dim=30):
        self.dim = dim
        self.bounds = [-600, 600]
        self.name = "Griewank"

    def evaluate(self, x):
        x = np.asarray(x)
        i = np.arange(1, self.dim + 1)
        sum_part = np.sum(x ** 2, axis=-1) / 4000
        prod_part = np.prod(np.cos(x / np.sqrt(i)), axis=-1)
        return 1 + sum_part - prod_part


class Schwefel:
    """f(x) = 418.9829*D - sum(x_i * sin(sqrt(|x_i|))), range [-500, 500]^D.
    Global minimum f(420.9687,...,420.9687) = 0.
    """

    def __init__(self, dim=30):
        self.dim = dim
        self.bounds = [-500, 500]
        self.name = "Schwefel"
        self._optimum = 420.9687 * np.ones(dim)

    def evaluate(self, x):
        x = np.asarray(x)
        return 418.9829 * self.dim - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=-1)


class Levy:
    """Levy function, range [-10, 10]^D, global min f(1,...,1) = 0."""

    def __init__(self, dim=30):
        self.dim = dim
        self.bounds = [-10, 10]
        self.name = "Levy"

    def evaluate(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        w = 1 + (x - 1) / 4
        a = np.sin(np.pi * w[:, 0]) ** 2
        b = np.sum((w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2), axis=-1)
        c = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)
        result = a + b + c
        if result.shape == (1,):
            return result.item()
        return result


class ExpandedScafferF6:
    """Expanded Scaffer's F6 function, range [-100, 100]^D, global min f(0)=0."""

    def __init__(self, dim=30):
        self.dim = dim
        self.bounds = [-100, 100]
        self.name = "ExpandedScafferF6"

    def evaluate(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x1, x2 = x[:, :-1], x[:, 1:]
        s = x1 ** 2 + x2 ** 2
        result = np.sum(0.5 + (np.sin(np.sqrt(s)) ** 2 - 0.5) / (1 + 0.001 * s) ** 2, axis=-1)
        if result.shape == (1,):
            return result.item()
        return result


def get_all_functions():
    """Return a list of all benchmark function classes."""
    return [Sphere, BentCigar, Zakharov, Rosenbrock, Rastrigin,
            Ackley, Griewank, Schwefel, Levy, ExpandedScafferF6]
