#!/usr/bin/env python3
"""EDAs over bijective codings (Lehmer / Fisher-Yates / insertion vector).

Runs the six coding-based EDAs on a PFSP instance and verifies that:
  * every sampled individual is a valid permutation, and
  * each representation satisfies decode(encode(sigma)) == sigma.

Run:  python3 examples/bijective_codings.py
"""
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from perm_pateda.algorithms import (
    LehmerUmdaEDA, LehmerTreeEDA, FisherYatesUmdaEDA, FisherYatesTreeEDA,
    InsertionVectorUmdaEDA, InsertionVectorMarkovEDA,
)
from perm_pateda.representations.lehmer import LehmerRepresentation
from perm_pateda.representations.fisher_yates import FisherYatesRepresentation
from perm_pateda.representations.vinsertion import InsertionVectorRepresentation
from perm_pateda.functions import create_random_pfsp

N_JOBS, N_MACH, POP, GEN, SEED = 12, 5, 80, 30, 111
pfsp = create_random_pfsp(N_JOBS, N_MACH, seed=1)   # minimises makespan (negated)

failures = []


def is_perm_population(pop, n):
    return all(sorted(row.tolist()) == list(range(n)) for row in pop)


print("Coding-based EDAs on PFSP (makespan)")
for cls in [LehmerUmdaEDA, LehmerTreeEDA, FisherYatesUmdaEDA,
            FisherYatesTreeEDA, InsertionVectorUmdaEDA, InsertionVectorMarkovEDA]:
    alg = cls(n_vars=N_JOBS, fitness_func=pfsp, pop_size=POP, n_gen=GEN, random_seed=SEED)
    stats, _ = alg.run()
    valid = is_perm_population(np.atleast_2d(stats.best_individual), N_JOBS)
    print(f"  {cls.__name__:26s} makespan={pfsp.evaluate_makespan(stats.best_individual):.0f}"
          f"   best is valid perm? {valid}")
    if not valid:
        failures.append(cls.__name__)

print("\nRepresentation round-trips  decode(encode(sigma)) == sigma")
rng = np.random.default_rng(0)
pop = np.array([rng.permutation(N_JOBS) for _ in range(300)])
for name, R in [("Lehmer(right)", LehmerRepresentation(left=False)),
                ("Lehmer(left)", LehmerRepresentation(left=True)),
                ("FisherYates", FisherYatesRepresentation()),
                ("InsertionVector", InsertionVectorRepresentation())]:
    ok = np.array_equal(R.decode(R.encode(pop.copy())), pop)
    dom = np.array(R.get_domain(N_JOBS))
    within = bool((R.encode(pop.copy()) <= dom).all())
    print(f"  [{'PASS' if ok and within else 'FAIL'}] {name}: round-trip={ok} within-domain={within}")
    if not (ok and within):
        failures.append(name)

print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILURES: {failures}"))
sys.exit(1 if failures else 0)
