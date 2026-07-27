#!/usr/bin/env python3
"""Statistical comparison utilities on a small multi-algorithm study.

Builds a results table by running several EDAs on several LOP instances with
several seeds, then exercises summary_table / friedman_test / wilcoxon_pairwise /
critical_difference_plot.  The helpers accept both the LONG format (used here)
and the WIDE format documented in the user guide (rows=instances, columns=algorithms);
both are demonstrated.

Run:  python3 examples/statistical_comparison.py
"""
import os
import tempfile
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from perm_pateda import (
    MallowsKendallEDA, EHMEDA, PlackettLuceEDA,
    summary_table, friedman_test, wilcoxon_pairwise, critical_difference_plot,
)
from perm_pateda.functions import create_random_lop

ALGS = {"MEDA-Kendall": MallowsKendallEDA, "EHM": EHMEDA, "PL": PlackettLuceEDA}
INSTANCES = [f"lop{i}" for i in range(3)]
SEEDS = [111, 222, 333]
N = 12

records = []
for i, inst in enumerate(INSTANCES):
    lop = create_random_lop(N, seed=i)
    for alg_name, cls in ALGS.items():
        for seed in SEEDS:
            stats, _ = cls(n_vars=N, fitness_func=lop, pop_size=50, n_gen=15,
                           random_seed=seed).run()
            records.append({"Problem": inst, "Algorithm": alg_name,
                            "Seed": seed, "Best Fitness": stats.best_fitness_overall})

long_df = pd.DataFrame(records)

print("=== summary_table (long input) ===")
print(summary_table(long_df).to_string())

print("\n=== friedman_test ===")
print(friedman_test(long_df))

print("\n=== wilcoxon_pairwise ===")
print(wilcoxon_pairwise(long_df).to_string(index=False))

# Wide format (UG Section 9.2): rows = instances, columns = algorithms, values = mean
wide_df = long_df.groupby(["Problem", "Algorithm"])["Best Fitness"].mean().unstack("Algorithm")
print("\n=== summary_table (WIDE input, auto-detected) ===")
print(summary_table(wide_df).to_string())

out = os.path.join(tempfile.gettempdir(), "critical_difference.png")
try:
    critical_difference_plot(long_df, alpha=0.05, filepath=out, maximize=True)
    print(f"\ncritical-difference diagram saved to {out}")
except Exception as exc:  # pragma: no cover - plotting is optional
    print(f"\n(critical-difference plot skipped: {exc})")
