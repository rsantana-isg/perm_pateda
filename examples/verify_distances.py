#!/usr/bin/env python3
"""Verify the permutation distances against the JSS worked examples and identities.

Checks (see paper/TOMS_JSS/jss_main.pdf, Section 3):
  * Kendall d_k(213645) = 3          (via the inversion vector V)
  * Cayley  d_c(213645) = 3          (n - #cycles)
  * Ulam    d_u(2136457) = 2         (n - LIS)
  * right-invariance  d(sigma, pi) == d(sigma pi^-1, e)   for all four distances
  * cayley_distance agrees with an independent n-#cycles count on random pairs

Run:  python3 examples/verify_distances.py
"""
import sys
import numpy as np

from perm_pateda.distances import (
    kendall_distance, cayley_distance, ulam_distance, hamming_distance,
    _v_vector, _x_vector_cycles,
)


def _compose_inverse(sigma, pi):
    """Return sigma . pi^-1 (apply pi^-1 then sigma)."""
    inv_pi = np.argsort(pi)
    return sigma[inv_pi]


def _n_minus_cycles(p, ref):
    """Cayley distance computed independently as n - (number of cycles of p.ref^-1)."""
    c = p[np.argsort(ref)]
    n = len(c)
    seen = np.zeros(n, dtype=bool)
    cycles = 0
    for i in range(n):
        if not seen[i]:
            cycles += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = c[j]
    return n - cycles


def main() -> int:
    failures = []

    def check(name, got, expected):
        ok = got == expected
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, expected {expected}")
        if not ok:
            failures.append(name)

    # JSS examples use 1-indexed permutations; convert to 0-indexed.
    print("JSS worked examples")
    s1 = np.array([2, 1, 3, 6, 4, 5]) - 1          # sigma = 213645
    ident6 = np.arange(6)
    check("Kendall d_k(213645)", kendall_distance(s1, ident6), 3)
    check("sum(V) == d_k", int(_v_vector(s1).sum()), 3)
    check("Cayley d_c(213645)", cayley_distance(s1, ident6), 3)
    check("sum(X) == d_c", int(_x_vector_cycles(s1).sum()), 3)

    s2 = np.array([2, 1, 3, 6, 4, 5, 7]) - 1        # sigma = 2136457 (LIS = 5)
    check("Ulam d_u(2136457)", ulam_distance(s2, np.arange(7)), 2)

    print("\nRight-invariance  d(sigma, pi) == d(sigma pi^-1, e)")
    rng = np.random.default_rng(0)
    n = 9
    e = np.arange(n)
    inv_ok = {"kendall": True, "cayley": True, "ulam": True, "hamming": True}
    for _ in range(1000):
        a, b = rng.permutation(n), rng.permutation(n)
        comp = _compose_inverse(a, b)
        if kendall_distance(a, b) != kendall_distance(comp, e):
            inv_ok["kendall"] = False
        if cayley_distance(a, b) != cayley_distance(comp, e):
            inv_ok["cayley"] = False
        if ulam_distance(a, b) != ulam_distance(comp, e):
            inv_ok["ulam"] = False
        if hamming_distance(a, b) != hamming_distance(comp, e):
            inv_ok["hamming"] = False
    for k, v in inv_ok.items():
        check(f"right-invariant ({k})", v, True)

    print("\ncayley_distance == independent n-#cycles on 2000 random pairs")
    agree = all(
        cayley_distance(a := rng.permutation(n), b := rng.permutation(n))
        == _n_minus_cycles(a, b)
        for _ in range(2000)
    )
    check("cayley == n-#cycles", agree, True)

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
