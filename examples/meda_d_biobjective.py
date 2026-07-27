#!/usr/bin/env python3
"""Decomposition-based multi-objective EDAs (MEDA/D) on a bi-objective problem.

Runs the nine MEDA/D models on a QAP + LOP bi-objective instance, reports the
Pareto-front size and the hypervolume, and additionally runs one MIXED
minimise/maximise case to exercise the corrected scalarization.

Run:  python3 examples/meda_d_biobjective.py
"""
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import perm_pateda.multiobjective as mo
from perm_pateda.functions import create_random_qap, create_random_lop

N, SEED = 15, 0
N_SUB, N_GEN = 30, 30
qap = create_random_qap(N, seed=1)
lop = create_random_lop(N, seed=2)


def hypervolume_2d(objs, ref):
    """Simple 2-D hypervolume for a MINIMISATION front (objs shape (k,2))."""
    if len(objs) == 0:
        return 0.0
    pts = objs[np.argsort(objs[:, 0])]
    hv, prev_x = 0.0, ref[0]
    # iterate from worst-x to best-x accumulating rectangles up to ref
    for x, y in pts[::-1]:
        if y < ref[1] and x < ref[0]:
            hv += (prev_x - x) * (ref[1] - y)
            prev_x = x
    return hv


MODELS = ["MEDA_D_MK", "MEDA_D_KENDALL", "MEDA_D_ULAM", "MEDA_D_GMKENDALL",
          "MEDA_D_GMCAYLEY", "MEDA_D_PLACKETT_LUCE", "MEDA_D_MIXTURE_PLACKETT_LUCE",
          "MEDA_D_NHM", "MEDA_D_EHM"]

# both objectives expressed as minimisation (negated maximise-style problems)
objs = [lambda p: -qap(p), lambda p: -lop(p)]
# reference (nadir-ish) point for HV, from a random baseline
rng = np.random.default_rng(0)
base = np.array([[-qap(rng.permutation(N)), -lop(rng.permutation(N))] for _ in range(200)])
ref = base.max(axis=0) * 1.05

print(f"Bi-objective QAP+LOP (both minimised), n={N}\n")
print(f"{'model':30s} {'front':>6s} {'hypervolume':>12s}")
for name in MODELS:
    cls = getattr(mo, name)
    res = cls(objectives=objs, n=N, n_subproblems=N_SUB, neighbourhood_size=10, nr=2,
              scalarization="tchebycheff", minimize=(True, True), seed=SEED
              ).run(n_generations=N_GEN)
    front = np.asarray(res["pareto_objectives"], dtype=float)
    hv = hypervolume_2d(front, ref) if front.size else 0.0
    print(f"{name:30s} {len(res['pareto_solutions']):6d} {hv:12.3e}")

print("\nMixed objectives (obj0 minimise, obj1 MAXIMISE) with MEDA_D_GMCAYLEY")
objs_mixed = [lambda p: -qap(p), lambda p: lop(p)]   # obj1 maximised directly
res = mo.MEDA_D_GMCAYLEY(objectives=objs_mixed, n=N, n_subproblems=N_SUB,
                         neighbourhood_size=10, minimize=(True, False), seed=SEED
                         ).run(n_generations=N_GEN)
print(f"  Pareto front size = {len(res['pareto_solutions'])}")
