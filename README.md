# 🧠 Intelligent Optimization Algorithm Study

A beginner-friendly repository for learning intelligent optimization algorithms, featuring 3 algorithms benchmarked on 10 test functions.

## 📁 Directory Structure

```
├── algorithm/              # Algorithm implementations
│   ├── pso.py              #   Particle Swarm Optimization
│   ├── ga.py               #   Genetic Algorithm
│   └── ssa.py              #   Sparrow Search Algorithm
├── cec2017/                # Benchmark functions (10)
│   ├── __init__.py
│   └── functions.py        #   Sphere, BentCigar, Zakharov,
│                           #   Rosenbrock, Rastrigin, Ackley,
│                           #   Griewank, Schwefel, Levy,
│                           #   ExpandedScafferF6
├── main.py                 # Entry point: batch run & comparison
├── plot.py                 # Convergence & summary plots
├── results/                # Output images + CSV
└── README.md
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install numpy matplotlib

# Run all experiments
python main.py
```

After running, `results/` will contain:
- `comparison_*.png` — per-function comparison of all 3 algorithms (mean ± std)
- `summary_all.png` — full grid summary across all functions
- `summary.csv` — final best-fitness mean & std for each combination

## 🔧 Customization

Edit the constants in `main.py`:

```python
DIM = 30          # Problem dimension
POP_SIZE = 50     # Population size
MAX_ITER = 200    # Max iterations
N_SEEDS = 5       # Random seeds (for error bands)
```

## 📊 Algorithms

| Algorithm | Tag | Family | Core Idea |
|-----------|-----|--------|-----------|
| Particle Swarm Optimization | PSO | Swarm Intelligence | Particles adjust velocity via personal & global best |
| Genetic Algorithm | GA | Evolutionary | Selection, crossover, mutation + elitism |
| Sparrow Search Algorithm | SSA | Swarm Intelligence | Producer-scrounger-vigilante cooperative foraging |

## 📈 Benchmark Functions

10 functions covering unimodal/multimodal and separable/non-separable landscapes. All have a global minimum of 0.

## 📝 License

MIT
