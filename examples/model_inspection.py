#!/usr/bin/env python3
"""Inspect the learned model of each family (user guide Section 9.1).

Every learner returns a plain dict, so the model is directly inspectable. This
script learns one model per family from a shared selected population and prints
the most informative quantities (consensus, theta, matrices, weights, ...).

Run:  python3 examples/model_inspection.py
"""
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from perm_pateda.functions import create_random_lop
from perm_pateda.learning.mallows import (
    LearnMallowsKendall, LearnMallowsCayley, LearnGeneralizedMallowsCayley,
)
from perm_pateda.learning.histogram import LearnEHM, LearnNHM
from perm_pateda.learning.plackett_luce import LearnPlackettLuce
from perm_pateda.learning.dsm import LearnDSM
from perm_pateda.learning.hamming_kmm import LearnHammingKMM
from perm_pateda.learning.lehmer import LearnLehmerUMDA

N = 12
lop = create_random_lop(N, seed=1)
rng = np.random.default_rng(0)
cand = np.array([rng.permutation(N) for _ in range(120)])
fit = np.array([lop(p) for p in cand])
sel = cand[np.argsort(fit)[::-1][:40]]           # 40 best permutations
zeros = np.zeros(len(sel))
card = np.arange(N)


print(f"Selected population: {len(sel)} permutations of length {N}\n")

m = LearnMallowsKendall().learn(0, N, card, sel, zeros)
print("Mallows-Kendall:")
print(f"  consensus = {m['consensus']}")
print(f"  theta     = {m['theta']:.3f}")

m = LearnMallowsCayley()(0, N, card, sel, zeros)
print("Mallows-Cayley:")
print(f"  theta     = {m['theta']:.3f}   (single spread parameter)")

m = LearnGeneralizedMallowsCayley()(0, N, card, sel, zeros)
print("GM-Cayley:")
print(f"  theta[:5] = {np.array2string(np.asarray(m['theta'])[:5], precision=2)}"
      f"   (per-position spreads)")

m = LearnEHM().learn(0, N, card, sel, zeros)
print("Edge histogram: matrix entropy (diversity proxy) = "
      f"{-np.nansum((r := m['ehm_matrix'] / m['ehm_matrix'].sum()) * np.log(r + 1e-12)):.3f}")

m = LearnNHM()(0, N, card, sel, zeros)
print(f"Node histogram: row 0 top-3 items = {list(np.argsort(-m['nhm_matrix'][0])[:3])}")

m = LearnPlackettLuce().learn(0, N, card, sel, zeros)
print(f"Plackett-Luce: implied ranking (best first) = {list(np.argsort(-m['weights']))[:6]} ...")

m = LearnDSM().learn(0, N, card, sel, zeros)
print(f"DSM: alpha = {m['alpha']:.4g}, matrix shape = {m['dsm'].shape}")

m = LearnHammingKMM(n_gen=50).learn(0, N, card, sel, zeros)
print(f"Hamming-KMM: theta = {m['theta']:.3f}, expected_dist = {m['expected_dist']:.3f}, "
      f"#centers = {len(m['centers'])}")

m = LearnLehmerUMDA().learn(0, N, card, sel, zeros)
print(f"Lehmer-UMDA: {len(m['marginals'])} per-position marginals, "
      f"domain sizes = {list(m['domain_sizes'])}")
