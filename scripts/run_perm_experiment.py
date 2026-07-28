#!/usr/bin/env python3
"""Run ONE permutation-EDA experiment (one seed x one algorithm x one instance).

This is the single-experiment runner invoked by ``slurm/slurm_perm_algs.sh`` on
the cluster.  It follows the project convention: positional arguments only, seed
first.  It loads one real benchmark instance (the same loaders used by
``scripts/compare_representations2.py``), runs one EDA once, and prints a
self-describing report to standard output (the SLURM wrapper redirects stdout to
a ``.dat`` file whose name encodes the parameters).

Positional arguments
--------------------
    1  seed              random seed (int)
    2  algorithm         algorithm label, e.g. Mallows-K, Lehmer-UMDA, EHM ...
                         (one of the labels in compare_representations.build_algorithms)
    3  problem           LOP | QAP | PFSP | TSP
    4  instance          instance name (without the extension), e.g. N-be75eec,
                         tai15a, tai50_5_0, burma14
    5  benchmark_dir     root directory holding LOP/ QAP/ PFSP/ TSP subfolders
    6  pop_size          population size (int)
    7  n_gen             number of generations (int)
    8  selection_ratio   truncation ratio (float)
    9  laplace_smoothing (optional) Laplace smoothing for the coding EDAs (default 0.01)

The report contains, for posterior analysis:
    * a metadata header (algorithm, problem, instance, n_vars, parameters, seed,
      objective direction/unit, wall-clock time, versions, timestamp),
    * the best (best-so-far) and mean fitness at EVERY generation in the problem's
      NATURAL objective units, plus the population diversity (normalized
      positional entropy),
    * the final best solution (0-indexed permutation) and its fitness.

Example
-------
    python3 scripts/run_perm_experiment.py 1 Mallows-K LOP N-be75eec Instances 400 60 0.5
"""
from __future__ import annotations

import argparse
import datetime as _dt
import platform
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

# Make the sibling scripts importable (loaders + algorithm registry).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_representations2 import _LOADERS          # noqa: E402
from compare_representations import build_algorithms    # noqa: E402
from evolution_tracking import run_with_history         # noqa: E402


def _positional_args(argv):
    if len(argv) < 8:
        sys.stderr.write(
            "Usage: run_perm_experiment.py SEED ALG PROBLEM INSTANCE BENCHMARK_DIR "
            "POP GEN SEL_RATIO [LAPLACE]\n"
        )
        raise SystemExit(2)
    seed = int(argv[0])
    algorithm = argv[1]
    problem = argv[2].upper()
    instance = argv[3]
    benchmark_dir = Path(argv[4])
    pop_size = int(argv[5])
    n_gen = int(argv[6])
    selection_ratio = float(argv[7])
    laplace = float(argv[8]) if len(argv) > 8 else 0.01
    return (seed, algorithm, problem, instance, benchmark_dir,
            pop_size, n_gen, selection_ratio, laplace)


def _load_instance(problem: str, instance: str, benchmark_dir: Path) -> dict:
    if problem not in _LOADERS:
        sys.stderr.write(f"Unknown problem '{problem}'. Use one of {list(_LOADERS)}.\n")
        raise SystemExit(2)
    _names, subdir, ext, loader = _LOADERS[problem]
    path = benchmark_dir / subdir / f"{instance}{ext}"
    if not path.exists():
        sys.stderr.write(f"Instance file not found: {path}\n")
        raise SystemExit(2)
    return loader(path, instance)


def _select_algorithm(label: str, laplace: float):
    algs = {lab: (cls, extra)
            for lab, cls, extra in build_algorithms(argparse.Namespace(laplace=laplace))}
    if label not in algs:
        sys.stderr.write(
            f"Unknown algorithm '{label}'. Available: {', '.join(algs)}\n")
        raise SystemExit(2)
    return algs[label]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    (seed, algorithm, problem, instance, benchmark_dir,
     pop_size, n_gen, selection_ratio, laplace) = _positional_args(argv)

    prob = _load_instance(problem, instance, benchmark_dir)
    cls, extra = _select_algorithm(algorithm, laplace)

    n_vars = prob["n_vars"]
    better = prob["better"]                    # "max" or "min"
    sign = 1.0 if better == "max" else -1.0    # internal fitness is always maximised
    maximize_natural = (better == "max")

    t0 = time.perf_counter()
    hist = run_with_history(
        cls, n_vars, prob["fitness"],
        pop_size=pop_size, n_gen=n_gen, selection_ratio=selection_ratio,
        random_seed=seed, extra_kwargs=extra,
    )
    elapsed = time.perf_counter() - t0

    # Convert internal (maximised) values to the problem's natural objective.
    best_internal = np.asarray(hist["best_fitness"], dtype=float)   # best-so-far
    mean_internal = np.asarray(hist["mean_fitness"], dtype=float)
    diversity = np.asarray(hist["diversity"], dtype=float)
    best_nat = sign * best_internal
    mean_nat = sign * mean_internal
    best_solution = hist["best_individual"]
    best_fitness_natural = sign * float(hist["best_fitness_overall"])

    out = sys.stdout
    w = out.write
    w("# perm_pateda single-experiment result\n")
    w(f"algorithm: {algorithm}\n")
    w(f"problem: {problem}\n")
    w(f"instance: {instance}\n")
    w(f"instance_label: {prob['name']}\n")
    w(f"n_vars: {n_vars}\n")
    w(f"pop_size: {pop_size}\n")
    w(f"n_gen: {n_gen}\n")
    w(f"selection_ratio: {selection_ratio}\n")
    w(f"laplace_smoothing: {laplace}\n")
    w(f"seed: {seed}\n")
    w(f"maximize: {maximize_natural}\n")            # objective direction (natural units)
    w(f"unit: {prob['unit']}\n")
    w(f"time_seconds: {elapsed:.6f}\n")
    w(f"n_records: {len(best_nat)}\n")               # = n_gen + 1 (initial + each gen)
    w(f"best_fitness_overall: {best_fitness_natural:.6f}\n")
    w(f"best_fitness_internal: {float(hist['best_fitness_overall']):.6f}\n")
    w(f"python: {platform.python_version()}\n")
    try:
        import perm_pateda
        w(f"perm_pateda_version: {getattr(perm_pateda, '__version__', 'unknown')}\n")
    except Exception:
        pass
    w(f"timestamp: {_dt.datetime.now().isoformat(timespec='seconds')}\n")

    # Per-generation evolution (natural objective units).  Column "best" is the
    # best-so-far fitness; "mean" is the population mean; "diversity" is the
    # normalized mean positional entropy in [0, 1].
    w("# --- per-generation evolution (natural units) ---\n")
    w("# gen best mean diversity\n")
    for g in range(len(best_nat)):
        w(f"{g} {best_nat[g]:.6f} {mean_nat[g]:.6f} {diversity[g]:.6f}\n")

    # Final best solution (0-indexed permutation).
    w("# --- best solution (0-indexed permutation) ---\n")
    w("best_solution: " + " ".join(str(int(x)) for x in best_solution) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
