#!/usr/bin/env python3
"""Compare permutation EDAs across many permutation problems and instances.

Algorithms (18)
---------------
1. Bijective-coding EDAs -- the full 3 x 3 grid:
       Lehmer / Fisher-Yates / insertion vector  x  UMDA / Tree (Chow-Liu) / Markov
2. EDAs with permutation-based probability distributions on the symmetric group:
       Mallows-Kendall, Mallows-Cayley, Mallows-Ulam,
       Generalized-Mallows-Kendall, Generalized-Mallows-Cayley,
       Plackett-Luce, edge histogram (EHM), node histogram (NHM),
       doubly stochastic matrix (DSM-AS).

Problems (7 types, up to 3 instances each)
------------------------------------------
    TSP, PFSP, LOP, QAP  and the permutation-picture graph problems
    MIS (maximum independent set), MaxCut, MVC (minimum vertex cover).
Each type contributes up to ``--max-instances`` instances of different sizes,
so the study spans a broad set of problems (21 instances by default).  All three
graph problems use the strict permutation-picture (prefix) decoder, so on large
sparse/dense instances they discriminate between the algorithms.

Output
------
* a results table per instance (natural objective units),
* an across-everything average-rank table + Friedman test,
* (with pandas) the library ``summary_table``,
* a boxplot per instance (results/compare_representations.png),
* if ``k >= 15``: a **critical-difference (CD) diagram per problem type**
  (results/cd/cd_<TYPE>.png, aggregating that type's instances x repetitions)
  plus an overall diagram (results/cd/cd_ALL.png).  In a CD diagram two
  algorithms are joined by a horizontal bar when they are NOT significantly
  different (their average ranks differ by less than the Nemenyi CD).

Parallelism
-----------
Every (instance, algorithm, repetition) is an independent EDA run.  Use
``--jobs N`` (up to your CPU count, e.g. 10) to run them in a process pool.
Results are identical regardless of ``--jobs`` (each run is seeded).

Usage
-----
    python3 scripts/compare_representations.py --k 15 --jobs 10           # overnight
    python3 scripts/compare_representations.py --k 15 --jobs 10 --pop 200 --gen 100
    python3 scripts/compare_representations.py --max-instances 1 --k 5 --jobs 4  # quick
    python3 scripts/compare_representations.py -h
"""
from __future__ import annotations

import argparse
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import friedmanchisquare, rankdata

warnings.filterwarnings("ignore")  # inherited by worker processes on import

from perm_pateda.functions import (
    create_random_tsp, create_random_pfsp, create_random_lop, create_random_qap,
    create_random_mis, create_random_max_cut, create_random_mvc,
)
# Bijective-coding EDAs (3 representations x 3 models)
from perm_pateda.algorithms import (
    LehmerUmdaEDA, LehmerTreeEDA, LehmerMarkovEDA,
    FisherYatesUmdaEDA, FisherYatesTreeEDA, FisherYatesMarkovEDA,
    InsertionVectorUmdaEDA, InsertionVectorTreeEDA, InsertionVectorMarkovEDA,
)
# EDAs with permutation-based probability distributions
from perm_pateda import (
    MallowsKendallEDA, MallowsCayleyEDA, MallowsUlamEDA,
    GMallowsKendallEDA, GMallowsCayleyEDA,
    PlackettLuceEDA, EHMEDA, NHMEDA, DSMASEDA,
)


# ---------------------------------------------------------------------------
# Problem instance specifications (up to 3 instances per problem type)
# ---------------------------------------------------------------------------
# For TSP/LOP/QAP/MIS/MaxCut/MVC the value is the permutation length n; for PFSP
# it is a (n_jobs, n_machines) pair.  Graph problems keep moderate sizes because
# their evaluation is more expensive (MaxCut is O(n^3) per solution).
PROBLEM_SPECS = {
    "TSP":    [15, 20, 25],
    "PFSP":   [(15, 5), (20, 5), (20, 10)],
    "LOP":    [15, 20, 25],
    "QAP":    [10, 12, 15],
    # Graph problems: sizes enlarged ~2-5x and per-type densities tuned so the
    # instances stop being trivially saturated (the small sparse defaults were
    # solved to optimality by every algorithm).  All three now use the strict
    # permutation-picture (prefix) decoder.  Empirically:
    #   * MaxCut discriminates on DENSE graphs (p=0.5);
    #   * MIS and MVC discriminate on SPARSE, large graphs (p=0.1): the prefix
    #     decoder must place a whole independent set / vertex cover contiguously,
    #     which is a genuinely hard ordering problem there.
    # MaxCut is kept a bit smaller because its evaluation is O(n^3) per solution.
    "MIS":    [60, 80, 100],
    "MaxCut": [30, 45, 60],
    "MVC":    [60, 80, 100],
}

# Edge probabilities for the random graph problems (tuned per type, see above).
_GRAPH_P = {"MIS": 0.1, "MaxCut": 0.5, "MVC": 0.1}


def _make_instance(ptype: str, cfg, seed: int) -> dict:
    """Create one problem instance descriptor."""
    if ptype == "TSP":
        n = cfg
        return dict(name=f"TSP-{n}", ptype=ptype, n_vars=n,
                    fitness=create_random_tsp(n, seed=seed), better="min", unit="tour length")
    if ptype == "PFSP":
        jobs, mach = cfg
        return dict(name=f"PFSP-{jobs}x{mach}", ptype=ptype, n_vars=jobs,
                    fitness=create_random_pfsp(jobs, mach, seed=seed), better="min", unit="makespan")
    if ptype == "LOP":
        n = cfg
        return dict(name=f"LOP-{n}", ptype=ptype, n_vars=n,
                    fitness=create_random_lop(n, seed=seed), better="max", unit="objective")
    if ptype == "QAP":
        n = cfg
        return dict(name=f"QAP-{n}", ptype=ptype, n_vars=n,
                    fitness=create_random_qap(n, seed=seed), better="min", unit="cost")
    if ptype == "MIS":
        n = cfg
        return dict(name=f"MIS-{n}", ptype=ptype, n_vars=n,
                    fitness=create_random_mis(n, edge_probability=_GRAPH_P["MIS"], seed=seed),
                    better="max", unit="set size")
    if ptype == "MaxCut":
        n = cfg
        return dict(name=f"MaxCut-{n}", ptype=ptype, n_vars=n,
                    fitness=create_random_max_cut(n, edge_probability=_GRAPH_P["MaxCut"], seed=seed),
                    better="max", unit="cut size")
    if ptype == "MVC":
        n = cfg
        return dict(name=f"MVC-{n}", ptype=ptype, n_vars=n,
                    fitness=create_random_mvc(n, edge_probability=_GRAPH_P["MVC"], seed=seed),
                    better="min", unit="cover size")
    raise ValueError(f"unknown problem type {ptype}")


def build_problems(args: argparse.Namespace) -> list:
    problems = []
    idx = 0
    for ptype, configs in PROBLEM_SPECS.items():
        for cfg in configs[: args.max_instances]:
            problems.append(_make_instance(ptype, cfg, args.instance_seed + idx))
            idx += 1
    return problems


def build_algorithms(args: argparse.Namespace) -> list:
    """Return a list of (label, EDA class, extra_kwargs) triples."""
    lap = {"laplace_smoothing": args.laplace}
    coding = [
        ("Lehmer-UMDA",   LehmerUmdaEDA,            lap),
        ("Lehmer-Tree",   LehmerTreeEDA,            lap),
        ("Lehmer-Markov", LehmerMarkovEDA,          lap),
        ("FY-UMDA",       FisherYatesUmdaEDA,       lap),
        ("FY-Tree",       FisherYatesTreeEDA,       lap),
        ("FY-Markov",     FisherYatesMarkovEDA,     lap),
        ("IV-UMDA",       InsertionVectorUmdaEDA,   lap),
        ("IV-Tree",       InsertionVectorTreeEDA,   lap),
        ("IV-Markov",     InsertionVectorMarkovEDA, lap),
    ]
    distribution = [
        ("Mallows-K", MallowsKendallEDA,  {}),
        ("Mallows-C", MallowsCayleyEDA,   {}),
        ("Mallows-U", MallowsUlamEDA,     {}),
        ("GM-K",      GMallowsKendallEDA, {}),
        ("GM-C",      GMallowsCayleyEDA,  {}),
        ("PL",        PlackettLuceEDA,    {}),
        ("EHM",       EHMEDA,             {}),
        ("NHM",       NHMEDA,             {}),
        ("DSM-AS",    DSMASEDA,           {}),
    ]
    return coding + distribution


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=10, help="repetitions per algorithm (default 10; CD diagrams at >=15)")
    p.add_argument("--pop", type=int, default=100, help="population size (default 100)")
    p.add_argument("--gen", type=int, default=60, help="number of generations (default 60)")
    p.add_argument("--selection-ratio", type=float, default=0.5, help="truncation ratio (default 0.5)")
    p.add_argument("--laplace", type=float, default=0.01, help="Laplace smoothing for coding EDAs (default 0.01)")
    p.add_argument("--max-instances", type=int, default=3, help="max instances per problem type, 1..3 (default 3)")
    p.add_argument("--jobs", type=int, default=1, help="parallel worker processes, e.g. 10 (default 1)")
    p.add_argument("--base-seed", type=int, default=100, help="first repetition seed (default 100)")
    p.add_argument("--instance-seed", type=int, default=1, help="base seed for the problem instances (default 1)")
    p.add_argument("--cd-alpha", type=float, default=0.05, help="significance level for the CD diagrams (default 0.05)")
    p.add_argument("--no-plot", action="store_true", help="disable the boxplot figure")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Worker (must be module-level so it is picklable for the process pool)
# ---------------------------------------------------------------------------

def _run_one(task):
    """Run a single EDA execution.  Returns (instance, label, rep, fitness, error)."""
    inst, n_vars, label, cls, extra, pop, gen, sr, seed, rep, fitness = task
    try:
        alg = cls(
            n_vars=n_vars, fitness_func=fitness, pop_size=pop, n_gen=gen,
            selection_ratio=sr, random_seed=seed, **extra,
        )
        stats, _ = alg.run()
        return (inst, label, rep, float(stats.best_fitness_overall), None)
    except Exception as exc:  # keep the batch alive if one run fails
        return (inst, label, rep, float("nan"), repr(exc))


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def run_experiment(args, problems, algorithms):
    labels = [label for label, _, _ in algorithms]
    better = {p["name"]: p["better"] for p in problems}
    results = {p["name"]: {l: {"fitness": [None] * args.k, "natural": [None] * args.k}
                          for l in labels}
               for p in problems}

    tasks = []
    for p in problems:
        for label, cls, extra in algorithms:
            for rep in range(args.k):
                tasks.append((p["name"], p["n_vars"], label, cls, extra,
                              args.pop, args.gen, args.selection_ratio,
                              args.base_seed + rep, rep, p["fitness"]))

    total = len(tasks)
    jobs = max(1, min(args.jobs, os.cpu_count() or 1))
    print(f"Running {total} EDA executions "
          f"({len(algorithms)} algorithms x {len(problems)} instances x k={args.k}; "
          f"pop={args.pop}, gen={args.gen}) on {jobs} process(es) ...\n")

    def store(res):
        inst, label, rep, fit, err = res
        if err is not None:
            print(f"  ! {inst} / {label} / rep{rep} FAILED: {err}", flush=True)
        results[inst][label]["fitness"][rep] = fit
        results[inst][label]["natural"][rep] = fit if better[inst] == "max" else -fit

    step = max(1, total // 50)
    done = 0
    t0 = time.perf_counter()
    if jobs == 1:
        for t in tasks:
            store(_run_one(t))
            done += 1
            if done % step == 0 or done == total:
                print(f"  [{done:5d}/{total}]  ({time.perf_counter()-t0:6.1f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for fut in as_completed([ex.submit(_run_one, t) for t in tasks]):
                store(fut.result())
                done += 1
                if done % step == 0 or done == total:
                    print(f"  [{done:5d}/{total}]  ({time.perf_counter()-t0:6.1f}s)", flush=True)

    print(f"\nFinished in {time.perf_counter() - t0:.1f}s\n")
    return results, labels


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def per_problem_tables(problems, results, labels):
    for prob in problems:
        name, better, unit = prob["name"], prob["better"], prob["unit"]
        arrow = "lower is better" if better == "min" else "higher is better"
        print(f"=== {name} ({unit}, {arrow}) ===")
        print(f"  {'algorithm':13s} {'mean':>12s} {'std':>10s} {'best':>12s} {'median':>12s}")
        best_fn = np.nanmin if better == "min" else np.nanmax
        rows = []
        for label in labels:
            vals = np.array(results[name][label]["natural"], dtype=float)
            n_ok = int(np.sum(~np.isnan(vals)))
            rows.append((label, np.nanmean(vals),
                         np.nanstd(vals, ddof=1) if n_ok > 1 else 0.0,
                         best_fn(vals), np.nanmedian(vals)))
        finite = [r[1] for r in rows if not math.isnan(r[1])]
        best_mean = (min if better == "min" else max)(finite) if finite else float("nan")
        for label, mean, std, best, med in rows:
            star = " *" if mean == best_mean else "  "
            print(f"  {label:13s} {mean:12.2f} {std:10.2f} {best:12.2f} {med:12.2f}{star}")
        print("  (* = best mean)\n")


def rank_table(problems, results, labels, k):
    blocks = []
    for prob in problems:
        for rep in range(k):
            row = [results[prob["name"]][label]["fitness"][rep] for label in labels]
            if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in row):
                continue
            blocks.append(row)
    if not blocks:
        print("=== Average rank: no complete blocks, skipping ===\n")
        return
    mat = np.asarray(blocks, dtype=float)
    ranks = np.vstack([rankdata(-row, method="average") for row in mat])  # higher fitness -> rank 1
    avg_rank = ranks.mean(axis=0)
    print(f"=== Average rank across all instances x repetitions ({len(blocks)} blocks, 1 = best) ===")
    for idx in np.argsort(avg_rank):
        print(f"  {labels[idx]:13s} {avg_rank[idx]:.3f}")
    try:
        stat, p = friedmanchisquare(*[mat[:, j] for j in range(mat.shape[1])])
        print(f"\n  Friedman chi^2 = {stat:.3f},  p-value = {p:.4g}  "
              f"({'significant' if p < 0.05 else 'not significant'} at alpha=0.05)")
    except Exception as exc:  # pragma: no cover
        print(f"\n  Friedman test skipped: {exc}")
    print()


def library_stats(problems, results, labels, k):
    try:
        import pandas as pd
        from perm_pateda import summary_table
    except Exception as exc:
        print(f"(pandas/stats utilities not available: {exc})")
        return
    records = []
    for prob in problems:
        for label in labels:
            for rep in range(k):
                fit = results[prob["name"]][label]["fitness"][rep]
                if fit is None or (isinstance(fit, float) and math.isnan(fit)):
                    continue
                records.append({"Problem": prob["name"], "Algorithm": label,
                                "Seed": rep, "Best Fitness": fit})
    df = pd.DataFrame(records)
    print("=== summary_table (maximised fitness; higher = better) ===")
    print(summary_table(df).to_string(), "\n")


def boxplot(problems, results, labels, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(boxplot skipped: {exc})")
        return
    n = len(problems)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 3.6), squeeze=False)
    axes = axes.ravel()
    for ax, prob in zip(axes, problems):
        data = [np.array(results[prob["name"]][label]["natural"], dtype=float) for label in labels]
        data = [d[~np.isnan(d)] for d in data]
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_title(f"{prob['name']} ({prob['unit']})", fontsize=9)
        ax.tick_params(axis="x", rotation=90, labelsize=6)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"boxplot saved to {os.path.abspath(path)}")


def critical_difference_diagrams(problems, results, labels, k, alpha, out_dir):
    """One CD diagram per problem TYPE (blocks = its instances x repetitions),
    plus one overall diagram across every instance x repetition."""
    try:
        import pandas as pd
        from perm_pateda import critical_difference_plot
    except Exception as exc:
        print(f"(CD diagrams skipped: {exc})")
        return
    os.makedirs(out_dir, exist_ok=True)

    def build_long(names):
        recs = []
        for name in names:
            for label in labels:
                for rep in range(k):
                    fit = results[name][label]["fitness"][rep]
                    if fit is None or (isinstance(fit, float) and math.isnan(fit)):
                        continue
                    # Problem x Seed forms the Nemenyi block; use instance x rep.
                    recs.append({"Problem": name, "Algorithm": label,
                                 "Seed": rep, "Best Fitness": fit})
        return pd.DataFrame(recs)

    by_type = {}
    for p in problems:
        by_type.setdefault(p["ptype"], []).append(p["name"])

    print(f"=== Critical-difference diagrams (alpha={alpha}) ===")
    for ptype, names in by_type.items():
        df = build_long(names)
        path = os.path.join(out_dir, f"cd_{ptype}.png")
        try:
            critical_difference_plot(df, alpha=alpha, filepath=path, maximize=True)
            print(f"  {ptype:7s} ({len(names)} instance(s)): {os.path.abspath(path)}")
        except Exception as exc:  # pragma: no cover
            print(f"  {ptype}: skipped ({exc})")

    df_all = build_long([p["name"] for p in problems])
    path_all = os.path.join(out_dir, "cd_ALL.png")
    try:
        critical_difference_plot(df_all, alpha=alpha, filepath=path_all, maximize=True)
        print(f"  {'ALL':7s} ({len(problems)} instances): {os.path.abspath(path_all)}")
    except Exception as exc:  # pragma: no cover
        print(f"  ALL: skipped ({exc})")
    print()


def main() -> int:
    args = parse_args()
    problems = build_problems(args)
    algorithms = build_algorithms(args)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(repo_root, "results")

    print("Permutation EDA comparison")
    print(f"  {len(algorithms)} algorithms (9 bijective-coding + 9 permutation-distribution models)")
    print(f"  {len(problems)} instances across {len(PROBLEM_SPECS)} problem types "
          f"(<= {args.max_instances} instances/type): "
          f"{', '.join(p['name'] for p in problems)}")
    print(f"  k={args.k} repetitions"
          f"{'  (>=15 -> per-type CD diagrams)' if args.k >= 15 else '  (CD diagrams need k>=15)'}\n")

    results, labels = run_experiment(args, problems, algorithms)
    per_problem_tables(problems, results, labels)
    rank_table(problems, results, labels, args.k)
    library_stats(problems, results, labels, args.k)

    if args.k >= 15:
        critical_difference_diagrams(problems, results, labels, args.k,
                                     args.cd_alpha, os.path.join(results_dir, "cd"))
    else:
        print(f"(critical-difference diagrams not produced: k={args.k} < 15)\n")

    if not args.no_plot:
        os.makedirs(results_dir, exist_ok=True)
        boxplot(problems, results, labels,
                os.path.join(results_dir, "compare_representations.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())