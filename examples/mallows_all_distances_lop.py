#!/usr/bin/env python3
"""Run the five distance-based EDAs on one LOP instance and sweep consensus methods.

Demonstrates:
  * MEDA-Kendall/Cayley/Ulam and GMEDA-Kendall/Cayley on the same instance,
  * the consensus_method argument now exposed by the plug-and-play wrappers
    ("borda", "setmedian", "best").

Run:  python3 examples/mallows_all_distances_lop.py
"""
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from perm_pateda import (
    MallowsKendallEDA, MallowsCayleyEDA, MallowsUlamEDA,
    GMallowsKendallEDA, GMallowsCayleyEDA,
)
from perm_pateda.functions import create_random_lop

N, POP, GEN, SEED = 20, 100, 40, 111
lop = create_random_lop(N, seed=1)

ALGS = {
    "MEDA-Kendall":  MallowsKendallEDA,
    "MEDA-Cayley":   MallowsCayleyEDA,
    "MEDA-Ulam":     MallowsUlamEDA,
    "GMEDA-Kendall": GMallowsKendallEDA,
    "GMEDA-Cayley":  GMallowsCayleyEDA,
}

print(f"LOP instance n={N}  (higher fitness is better)\n")
print(f"{'algorithm':16s} {'borda':>10s} {'setmedian':>10s} {'best':>10s}")
for name, cls in ALGS.items():
    row = []
    for cm in ("borda", "setmedian", "best"):
        alg = cls(n_vars=N, fitness_func=lop, pop_size=POP, n_gen=GEN,
                  selection_ratio=0.3, random_seed=SEED, consensus_method=cm)
        stats, _ = alg.run()
        row.append(stats.best_fitness_overall)
    print(f"{name:16s} {row[0]:10.0f} {row[1]:10.0f} {row[2]:10.0f}")

print("\nDone. Each column is the best LOP objective found under that consensus estimator.")
