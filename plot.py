"""
Plotting utilities — all convergence data is shaped (n_runs, n_iters).
Plots show the mean curve with a ±1 std error band.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(data, title="Convergence Curve"):
    """Plot mean ± std from multi-run data of shape (n_runs, n_iters)."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    iters = np.arange(len(mean))

    plt.figure(figsize=(8, 5))
    plt.plot(iters, mean, linewidth=1.5)
    plt.fill_between(iters, mean - std, mean + std, alpha=0.25)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def plot_comparison(results, func_name):
    """
    Plot mean ± std convergence for multiple algorithms on one function.

    results: dict mapping algo_name -> 2D array (n_runs, n_iters)
    """
    plt.figure(figsize=(8, 5))
    iters = np.arange(results[list(results.keys())[0]].shape[1])
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for idx, (algo_name, data) in enumerate(results.items()):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        color = colors[idx % len(colors)]
        plt.plot(iters, mean, linewidth=1.5, color=color, label=algo_name)
        plt.fill_between(iters, mean - std, mean + std, alpha=0.2, color=color)

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title(f"Convergence Comparison on {func_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def plot_all_functions(all_results):
    """
    Grid plot: rows = functions, cols = algorithms.
    Each subplot shows mean ± std over multiple runs.

    all_results: dict[func_name][algo_name] -> 2D array (n_runs, n_iters)
    """
    func_names = list(all_results.keys())
    algo_names = list(all_results[func_names[0]].keys())
    n_funcs = len(func_names)
    n_algos = len(algo_names)

    _, axes = plt.subplots(n_funcs, n_algos, figsize=(4 * n_algos, 3 * n_funcs))
    if n_funcs == 1 and n_algos == 1:
        axes = np.array([[axes]])
    elif n_funcs == 1:
        axes = np.array([axes])
    elif n_algos == 1:
        axes = axes.reshape(-1, 1)

    for i, func_name in enumerate(func_names):
        for j, algo_name in enumerate(algo_names):
            ax = axes[i][j]
            data = all_results[func_name][algo_name]
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            iters = np.arange(len(mean))
            ax.plot(iters, mean, linewidth=1, color="steelblue")
            ax.fill_between(iters, mean - std, mean + std, alpha=0.2, color="steelblue")
            ax.set_yscale("log")
            ax.set_title(f"{algo_name} on {func_name}", fontsize=9)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fitness")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return plt
