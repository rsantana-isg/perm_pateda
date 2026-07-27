#!/usr/bin/env python3
"""Random-key EDAs on a TSP, illustrating the diminishing/cooling flags.

The three RK-EDAs search permutations through a continuous relaxation in [0,1]^n
using the continuous learners of pateda.  This script shows the effect of the
`diminishing` (rank rescaling) and `cooling` (scheduled sigma) flags.

Run:  python3 examples/random_key_edas.py
"""
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from perm_pateda import RKGaussianUMDAEDA, RKGaussianFullEDA, RKCopulaVinesEDA
from perm_pateda.functions import create_random_tsp

N, POP, GEN, SEED = 15, 100, 50, 111
tsp = create_random_tsp(N, seed=1)

sigma0 = 1.0 / (np.pi * np.log10(N))
print(f"TSP n={N};  RK-EDA cooling start sigma0 = 1/(pi*log10(n)) = {sigma0:.4f}\n")

ALGS = {
    "RK-Gaussian-UMDA": RKGaussianUMDAEDA,
    "RK-Gaussian-Full": RKGaussianFullEDA,
    "RK-Copula-Vines":  RKCopulaVinesEDA,
}

print(f"{'algorithm':20s} {'plain':>10s} {'dimin+cool':>12s}")
for name, cls in ALGS.items():
    plain = cls(n_vars=N, fitness_func=tsp, pop_size=POP, n_gen=GEN, random_seed=SEED,
                diminishing=False, cooling=False).run()[0].best_fitness_overall
    tuned = cls(n_vars=N, fitness_func=tsp, pop_size=POP, n_gen=GEN, random_seed=SEED,
                diminishing=True, cooling=True).run()[0].best_fitness_overall
    print(f"{name:20s} {-plain:10.1f} {-tuned:12.1f}")

print("\nColumns are best tour length (lower is better).")
