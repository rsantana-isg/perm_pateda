#!/usr/bin/env python3
"""Two ways to assemble a permutation EDA: component path vs. plug-and-play wrapper.

Reproduces user guide Section 3.3: build an EHM/TSP EDA once through the generic
pateda executor (EDAComponents + pateda.core.eda.EDA) and once through the
EHMEDA wrapper, and report both results.

Run:  python3 examples/component_path_vs_wrapper.py
"""
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from pateda.core.eda import EDA
from pateda.core.components import EDAComponents
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations

from perm_pateda.seeding import PermutationInit
from perm_pateda.learning.histogram import LearnEHM
from perm_pateda.sampling.histogram import SampleEHM
from perm_pateda import EHMEDA
from perm_pateda.functions import create_random_tsp

N, POP, GEN, SEED = 15, 80, 40, 42
tsp = create_random_tsp(N, seed=SEED)

# --- component path ----------------------------------------------------------
components = EDAComponents(
    seeding=PermutationInit(),
    selection=TruncationSelection(ratio=0.5),
    learning=LearnEHM(),
    sampling=SampleEHM(),
    replacement=ElitistReplacement(n_elite=int(POP * 0.1)),
    stop_condition=MaxGenerations(GEN),
)
components.learning_params = {"symmetric": True, "beta_ratio": 0.01}
components.sampling_params = {"sample_size": POP}

eda = EDA(n_vars=N, cardinality=np.full(N, N, dtype=int), fitness_func=tsp,
          pop_size=POP, components=components, random_seed=SEED)
stats_c, _ = eda.run()
print(f"component path : best tour length = {-stats_c.best_fitness_overall:.2f}")

# --- wrapper path ------------------------------------------------------------
stats_w, _ = EHMEDA(n_vars=N, fitness_func=tsp, pop_size=POP, n_gen=GEN,
                    selection_ratio=0.5, random_seed=SEED).run()
print(f"wrapper path   : best tour length = {-stats_w.best_fitness_overall:.2f}")

print("\nBoth paths assemble the same EHM/TSP EDA; small differences come from the "
      "wrapper's self-contained loop vs. the generic pateda executor.")
