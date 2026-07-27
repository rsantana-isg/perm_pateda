#!/usr/bin/env python3
"""Verify the V (Kendall) and X (Cayley) decomposition vectors and their inversion.

Checks:
  * sum(V(sigma)) == kendall_distance(sigma, e)
  * sum(X(sigma)) == cayley_distance(sigma, e)
  * V ranges: 0 <= V[j] <= n-1-j
  * V -> permutation -> V round-trip  (SampleMallowsKendall._generate_perm_from_v)
  * X -> permutation round-trip: _generate_perm_from_x(x) has X-vector == x

Run:  python3 examples/verify_decomposition_vectors.py
"""
import sys
import numpy as np

from perm_pateda.distances import (
    kendall_distance, cayley_distance, _v_vector, _x_vector_cycles,
    _generate_perm_from_x,
)
from perm_pateda.sampling.mallows import SampleMallowsKendall


def main() -> int:
    rng = np.random.default_rng(1)
    n = 8
    e = np.arange(n)
    failures = []
    v_helper = SampleMallowsKendall()._generate_perm_from_v

    n_pop = 500
    sums_ok = ranges_ok = v_round_ok = x_round_ok = True
    for _ in range(n_pop):
        p = rng.permutation(n)
        V = _v_vector(p)
        X = _x_vector_cycles(p)

        if int(V.sum()) != kendall_distance(p, e):
            sums_ok = False
        if int(X.sum()) != cayley_distance(p, e):
            sums_ok = False
        if not all(0 <= V[j] <= n - 1 - j for j in range(n)):
            ranges_ok = False

        # V -> perm -> V round-trip (build a valid V vector first)
        Vrand = np.array([rng.integers(0, n - j) for j in range(n)], dtype=int)
        Vrand[-1] = 0
        perm_from_v = v_helper(Vrand, n)
        if not np.array_equal(_v_vector(perm_from_v), Vrand):
            v_round_ok = False

        # X -> perm round-trip: the generated permutation must have X-vector == X
        Xrand = (rng.random(n - 1) < 0.5).astype(int)
        perm_from_x = _generate_perm_from_x(Xrand, n, rng)
        if not np.array_equal(_x_vector_cycles(perm_from_x), Xrand):
            x_round_ok = False

    for name, ok in [
        ("sum(V)==d_k and sum(X)==d_c", sums_ok),
        ("V within [0, n-1-j]", ranges_ok),
        ("V -> perm -> V round-trip", v_round_ok),
        ("X -> perm has matching X-vector", x_round_ok),
    ]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failures.append(name)

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
