#!/usr/bin/env python3
"""Plackett-Luce and mixture-of-Plackett-Luce EDAs, with model inspection.

Runs PL-EDA and MPL-EDA on a LOP instance and prints the learned item weights
(PL) and the mixing proportions + per-component weights (mixture).

Run:  python3 examples/plackett_luce_and_mixture.py
"""
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from perm_pateda import PlackettLuceEDA, PlackettLuceMixtureEDA
from perm_pateda.learning.plackett_luce import LearnPlackettLuce
from perm_pateda.learning.mixture_plackett_luce import LearnPlackettLuceMixture
from perm_pateda.functions import create_random_lop

N, POP, GEN, SEED = 15, 100, 40, 7
lop = create_random_lop(N, seed=1)

pl = PlackettLuceEDA(n_vars=N, fitness_func=lop, pop_size=POP, n_gen=GEN, random_seed=SEED)
s_pl, _ = pl.run()
print(f"PL-EDA        best LOP objective = {s_pl.best_fitness_overall:.0f}")

mpl = PlackettLuceMixtureEDA(n_vars=N, fitness_func=lop, pop_size=POP, n_gen=GEN,
                             random_seed=SEED, n_components=2)
s_mpl, _ = mpl.run()
print(f"MPL-EDA (K=2) best LOP objective = {s_mpl.best_fitness_overall:.0f}")

# --- inspect the learned models on the final selected population -------------
print("\nModel inspection (learned from a sample of good solutions)")
rng = np.random.default_rng(0)
# take a set of near-optimal-ish permutations by sorting a few random ones
cand = np.array([rng.permutation(N) for _ in range(POP)])
fit = np.array([lop(p) for p in cand])
sel = cand[np.argsort(fit)[::-1][:40]]

pl_model = LearnPlackettLuce().learn(0, N, np.arange(N), sel, np.zeros(len(sel)))
w = pl_model["weights"]
print("  PL item weights (higher = ranked earlier):")
print("   ", np.array2string(w, precision=3, suppress_small=True))
print("    implied ranking (items, best first):", list(np.argsort(-w)))

mpl_model = LearnPlackettLuceMixture(n_components=2, random_state=0).learn(
    0, N, np.arange(N), sel, np.zeros(len(sel)))
print(f"  Mixture mixing proportions (beta): "
      f"{np.array2string(mpl_model['mixing_weights'], precision=3)}")
for k, wk in enumerate(mpl_model["weights_per_component"]):
    print(f"    component {k} top-5 items: {list(np.argsort(-wk)[:5])}")
