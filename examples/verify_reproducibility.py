#!/usr/bin/env python3
"""Verify that every EDA is reproducible: same random_seed -> identical result.

This is the regression guard for the Cayley-sampler RNG fix (the Cayley-based
algorithms previously used the global NumPy RNG and were NOT reproducible).

Run:  python3 examples/verify_reproducibility.py
"""
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import perm_pateda as pp
import perm_pateda.multiobjective as mo
from perm_pateda.algorithms import (
    LehmerUmdaEDA, LehmerTreeEDA, FisherYatesUmdaEDA, FisherYatesTreeEDA,
    InsertionVectorUmdaEDA, InsertionVectorMarkovEDA,
)
from perm_pateda.functions import create_random_lop, create_random_qap


def _run_twice(make):
    a = make(); b = make()
    sa, _ = a.run(); sb, _ = b.run()
    return (sa.best_fitness_overall == sb.best_fitness_overall
            and np.array_equal(sa.best_individual, sb.best_individual))


def main() -> int:
    lop = create_random_lop(12, seed=0)
    failures = []

    single = [
        "MallowsKendallEDA", "MallowsCayleyEDA", "MallowsUlamEDA",
        "GMallowsKendallEDA", "GMallowsCayleyEDA", "HammingKMMEDA",
        "EHMEDA", "NHMEDA", "PlackettLuceEDA", "PlackettLuceMixtureEDA",
        "DSMPSEDA", "DSMASEDA", "RKGaussianUMDAEDA", "RKGaussianFullEDA",
        "RKCopulaVinesEDA",
    ]
    print("Single-objective EDAs")
    for name in single:
        cls = getattr(pp, name)
        ok = _run_twice(lambda cls=cls: cls(n_vars=12, fitness_func=lop,
                                            pop_size=40, n_gen=8, random_seed=7))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failures.append(name)

    print("Bijective-coding EDAs")
    for cls in [LehmerUmdaEDA, LehmerTreeEDA, FisherYatesUmdaEDA,
                FisherYatesTreeEDA, InsertionVectorUmdaEDA, InsertionVectorMarkovEDA]:
        ok = _run_twice(lambda cls=cls: cls(n_vars=12, fitness_func=lop,
                                            pop_size=40, n_gen=8, random_seed=7))
        print(f"  [{'PASS' if ok else 'FAIL'}] {cls.__name__}")
        if not ok:
            failures.append(cls.__name__)

    print("MEDA/D (multi-objective)")
    n = 12
    qap = create_random_qap(n, seed=1); lop2 = create_random_lop(n, seed=2)
    objs = [lambda p: -qap(p), lambda p: -lop2(p)]
    for name in ["MEDA_D_MK", "MEDA_D_KENDALL", "MEDA_D_ULAM", "MEDA_D_GMKENDALL",
                 "MEDA_D_GMCAYLEY", "MEDA_D_PLACKETT_LUCE",
                 "MEDA_D_MIXTURE_PLACKETT_LUCE", "MEDA_D_NHM", "MEDA_D_EHM"]:
        cls = getattr(mo, name)
        r1 = cls(objectives=objs, n=n, n_subproblems=15, neighbourhood_size=5,
                 minimize=(True, True), seed=0).run(n_generations=8)["pareto_objectives"]
        r2 = cls(objectives=objs, n=n, n_subproblems=15, neighbourhood_size=5,
                 minimize=(True, True), seed=0).run(n_generations=8)["pareto_objectives"]
        ok = len(r1) == len(r2) and np.array_equal(np.sort(r1, axis=0), np.sort(r2, axis=0))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failures.append(name)

    print("\n" + ("ALL REPRODUCIBLE" if not failures else f"NOT REPRODUCIBLE: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
