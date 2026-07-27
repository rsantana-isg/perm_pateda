#!/usr/bin/env python3
"""Kernels of Mallows models (Hamming) on a QAP, showing the E[K]/theta schedule.

The KMM learner derives theta from an exponentially decaying expected Hamming
distance E(t) (user guide Eq. 8-9).  This script runs HammingKMMEDA on a QAP and
also plots the scheduled expected distance and the derived theta across
generations (figure saved to the system temp dir; skipped if matplotlib absent).

Run:  python3 examples/hamming_kmm_qap.py
"""
import os
import tempfile
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from perm_pateda import HammingKMMEDA
from perm_pateda.learning.hamming_kmm import LearnHammingKMM
from perm_pateda.functions import create_random_qap

N, POP, GEN, SEED = 15, 120, 60, 111
qap = create_random_qap(N, seed=1)

# --- optimisation run --------------------------------------------------------
alg = HammingKMMEDA(n_vars=N, fitness_func=qap, pop_size=POP, n_gen=GEN,
                    selection_ratio=0.1, random_seed=SEED)
stats, _ = alg.run()
print(f"HammingKMM on QAP(n={N}): best cost = {-stats.best_fitness_overall:.1f}")

# --- inspect the exploration->exploitation schedule --------------------------
learner = LearnHammingKMM(n_gen=GEN)
rng = np.random.default_rng(0)
pop = np.array([rng.permutation(N) for _ in range(POP // 10)])
gens, e_sched, thetas = [], [], []
for g in range(GEN):
    m = learner(generation=g, n_vars=N, cardinality=np.arange(N),
                population=pop, fitness=np.zeros(len(pop)))
    gens.append(g); e_sched.append(m["expected_dist"]); thetas.append(m["theta"])

print(f"expected distance: {e_sched[0]:.2f} (gen 0) -> {e_sched[-1]:.2f} (gen {GEN-1})")
print(f"theta:             {thetas[0]:.2f} (gen 0) -> {thetas[-1]:.2f} (gen {GEN-1})")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(gens, e_sched, "b-o", ms=3, label="E[K] (scheduled)")
    ax1.set_xlabel("generation"); ax1.set_ylabel("expected Hamming distance", color="b")
    ax2 = ax1.twinx()
    ax2.plot(gens, thetas, "r-s", ms=3, label="theta")
    ax2.set_ylabel("theta", color="r")
    out = os.path.join(tempfile.gettempdir(), "hamming_kmm_schedule.png")
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"schedule figure saved to {out}")
except Exception as exc:  # pragma: no cover - plotting is optional
    print(f"(plot skipped: {exc})")
