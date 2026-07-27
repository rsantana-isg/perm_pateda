#!/usr/bin/env python3
"""Doubly stochastic matrix EDAs: probabilistic (PS) vs algebraic (AS) sampling.

Compares DSM-PS and DSM-AS on a QAP (quality and wall-clock), and checks that the
learned matrix is doubly stochastic (all rows and columns sum to one).

Run:  python3 examples/dsm_ps_vs_as.py
"""
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from perm_pateda import DSMPSEDA, DSMASEDA
from perm_pateda.learning.dsm import LearnDSM
from perm_pateda.functions import create_random_qap

N, POP, GEN, SEED = 15, 120, 40, 111
qap = create_random_qap(N, seed=1)

for name, cls in [("DSM-PS", DSMPSEDA), ("DSM-AS", DSMASEDA)]:
    t0 = time.perf_counter()
    stats, _ = cls(n_vars=N, fitness_func=qap, pop_size=POP, n_gen=GEN,
                   selection_ratio=0.3, random_seed=SEED).run()
    dt = time.perf_counter() - t0
    print(f"{name}: best QAP cost = {-stats.best_fitness_overall:.1f}   ({dt:.2f}s)")

# --- verify double stochasticity of the learned DSM --------------------------
rng = np.random.default_rng(0)
sel = np.array([rng.permutation(N) for _ in range(40)])
model = LearnDSM().learn(0, N, np.arange(N), sel, np.zeros(len(sel)))
D = model["dsm"]
row_ok = np.allclose(D.sum(axis=1), 1.0)
col_ok = np.allclose(D.sum(axis=0), 1.0)
nonneg = bool((D >= 0).all())
print(f"\nlearned DSM (alpha={model['alpha']:.4g}): "
      f"rows sum to 1? {row_ok}   cols sum to 1? {col_ok}   non-negative? {nonneg}")
