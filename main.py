"""
Main entry point: run all algorithms on all benchmark functions with 5 random seeds
and plot mean ± std convergence curves.
"""

import os
import csv
import numpy as np

from cec2017.functions import get_all_functions
from algorithm.pso import PSO
from algorithm.ga import GA
from algorithm.ssa import SSA
from plot import plot_comparison, plot_all_functions

RESULTS_DIR = "results"
DIM = 30
POP_SIZE = 50
MAX_ITER = 200
N_SEEDS = 5
BASE_SEED = 42


def run_multi_seed(algo_class, algo_name, func):
    """Run one algorithm on one function with N_SEEDS different seeds.
    Returns a 2D array of shape (N_SEEDS, MAX_ITER+1)."""
    curves = []
    for s in range(N_SEEDS):
        np.random.seed(BASE_SEED + s)
        algo = algo_class(func, pop_size=POP_SIZE, max_iter=MAX_ITER)
        _, _, convergence = algo.optimize()
        curves.append(convergence)
    curves = np.array(curves)
    mean_final = np.mean(curves[:, -1])
    std_final = np.std(curves[:, -1])
    print(f"  {algo_name:>6} on {func.name:<20}  best={mean_final:.4e} ± {std_final:.2e}")
    return curves


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    function_classes = get_all_functions()
    algorithms = [
        (PSO, "PSO"),
        (GA,  "GA"),
        (SSA, "SSA"),
    ]

    all_results = {}

    print("=" * 60)
    print(f"Running {len(algorithms)} algorithms on {len(function_classes)} functions")
    print(f"Dimension={DIM}, PopSize={POP_SIZE}, MaxIter={MAX_ITER}, Seeds={N_SEEDS}")
    print("=" * 60)

    for FuncClass in function_classes:
        func = FuncClass(dim=DIM)
        print(f"\n--- {func.name} ---")
        all_results[func.name] = {}

        for algo_class, algo_name in algorithms:
            all_results[func.name][algo_name] = run_multi_seed(
                algo_class, algo_name, func
            )

    # --- Per-function comparison plots ---
    for FuncClass in function_classes:
        func_name = FuncClass.__name__
        plot_comparison(all_results[func_name], func_name)
        import matplotlib.pyplot as plt
        plt.savefig(os.path.join(RESULTS_DIR, f"comparison_{func_name}.png"), dpi=150)
        plt.close()
        print(f"Saved results/comparison_{func_name}.png")

    # --- Summary grid ---
    plot_all_functions(all_results)
    import matplotlib.pyplot as plt
    plt.savefig(os.path.join(RESULTS_DIR, "summary_all.png"), dpi=200)
    plt.close()
    print("Saved results/summary_all.png")

    # --- Save summary CSV ---
    algo_names = [name for _, name in algorithms]
    header = ["Function"] + [f"{name}_mean" for name in algo_names] + [f"{name}_std" for name in algo_names]
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for func_name in all_results:
            row = [func_name]
            for algo_name in algo_names:
                data = all_results[func_name][algo_name]
                row.append(np.mean(data[:, -1]))
            for algo_name in algo_names:
                data = all_results[func_name][algo_name]
                row.append(np.std(data[:, -1]))
            writer.writerow(row)
    print(f"Saved results/summary.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
