#!/usr/bin/env python3
"""Verify Mallows / Generalized-Mallows learn -> sample round-trips.

For each distance-based model we build a population concentrated around the
identity, learn the model, draw a large sample, and check that the sampled mean
distance-to-consensus matches the selected mean (the model reproduces the data),
and that theta increases as the population gets more concentrated.

This is the regression guard for the GM-Cayley marginal / theta fix.

Run:  python3 examples/verify_mallows_roundtrip.py
"""
import sys
import numpy as np

from perm_pateda.distances import kendall_distance, cayley_distance, ulam_distance
from perm_pateda.learning.mallows import (
    LearnMallowsKendall, LearnMallowsCayley, LearnMallowsUlam,
    LearnGeneralizedMallowsKendall, LearnGeneralizedMallowsCayley,
)
from perm_pateda.sampling.mallows import (
    SampleMallowsKendall, SampleMallowsCayley, SampleMallowsUlam,
    SampleGeneralizedMallowsKendall, SampleGeneralizedMallowsCayley,
)

MODELS = {
    "Mallows-Kendall":   (LearnMallowsKendall(),  SampleMallowsKendall(),  kendall_distance),
    "Mallows-Cayley":    (LearnMallowsCayley(),   SampleMallowsCayley(),   cayley_distance),
    "Mallows-Ulam":      (LearnMallowsUlam(),     SampleMallowsUlam(),     ulam_distance),
    "GM-Kendall":        (LearnGeneralizedMallowsKendall(), SampleGeneralizedMallowsKendall(), kendall_distance),
    "GM-Cayley":         (LearnGeneralizedMallowsCayley(),  SampleGeneralizedMallowsCayley(),  cayley_distance),
}


def _sample(sampler, n, model, pop, size, rng):
    """Call sampler via .sample() when available, else __call__."""
    card = np.arange(n)
    fit = np.zeros(len(pop))
    if hasattr(sampler, "sample"):
        return sampler.sample(n_vars=n, model=model, cardinality=card,
                              population=pop, fitness=fit, sample_size=size, rng=rng)
    return sampler(n, model, card, pop, fit, size, rng)


def make_population(n, m, p_random, rng):
    base = np.arange(n)
    return np.array([rng.permutation(n) if rng.random() < p_random else base.copy()
                     for _ in range(m)])


def main() -> int:
    n, m = 8, 300
    consensus = np.arange(n)
    failures = []

    print(f"{'model':16s} {'sel.mean':>9s} {'samp.mean':>10s}  match?")
    for name, (learner, sampler, dist) in MODELS.items():
        rng = np.random.default_rng(0)
        pop = make_population(n, m, 0.3, rng)
        sel_mean = float(np.mean([dist(p, consensus) for p in pop]))

        model = learner(generation=0, n_vars=n, cardinality=np.arange(n),
                        selected_pop=pop, selected_fitness=np.zeros(m),
                        consensus_method="best")
        model["consensus"] = consensus.copy()   # fix consensus for a clean comparison

        samp = _sample(sampler, n, model, pop, 3000, np.random.default_rng(1))
        samp_mean = float(np.mean([dist(p, consensus) for p in samp]))

        # tolerance: distances are integers; allow 20% relative slack
        ok = abs(samp_mean - sel_mean) <= max(0.5, 0.20 * max(sel_mean, 1e-9))
        print(f"{name:16s} {sel_mean:9.3f} {samp_mean:10.3f}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)

    # theta monotonicity: more concentrated population -> larger theta (Kendall)
    print("\ntheta grows with concentration (Mallows-Kendall)")
    thetas = []
    for p_random in (0.9, 0.5, 0.1):
        rng = np.random.default_rng(3)
        pop = make_population(n, m, p_random, rng)
        model = LearnMallowsKendall()(0, n, np.arange(n), pop, np.zeros(m))
        thetas.append(model["theta"])
        print(f"  p_random={p_random:.1f}  theta={model['theta']:.3f}")
    mono = thetas[0] <= thetas[1] <= thetas[2]
    print(f"  [{'PASS' if mono else 'FAIL'}] theta non-decreasing as population concentrates")
    if not mono:
        failures.append("theta-monotonicity")

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
