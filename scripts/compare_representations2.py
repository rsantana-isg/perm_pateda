#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from perm_pateda.utils.benchmark_parsers import (
    parse_lolib, parse_qaplib, parse_tsplib,
)
from perm_pateda.functions.lop import load_lolib_instance
from perm_pateda.functions.pfsp import load_taillard_instance
from perm_pateda.functions.qap import load_qaplib_instance
from perm_pateda.functions.tsp import create_tsp_from_coordinates


sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_representations import (  # noqa: E402
    build_algorithms, run_experiment, per_problem_tables, rank_table,
    library_stats, boxplot, critical_difference_diagrams,
)
from evolution_tracking import track_evolution  # noqa: E402


LOP_INSTANCES = [
    "N-t59b11xx", "N-t59d11xx", "N-t59f11xx", "N-be75eec", "N-be75np",
    "N-be75oi", "N-be75tot", "N-tiw56r58", "N-tiw56r66", "N-tiw56r67",
    "N-tiw56r72", "N-stabu70", "N-stabu74", "N-stabu75", "N-usa79",
    "N-t65d11xx_100", "N-t65f11xx_100", "N-t65i11xx_100", "N-t65l11xx_100",
    "N-t65n11xx_100", "N-t65w11xx_110", "N-t69r11xx_110", "N-t70b11xx_110",
    "N-t70d11xx_110", "N-t70d11xxb_110", "N-t70f11xx_120", "N-t70i11xx_120",
    "N-t70k11xx_120", "N-t70l11xx_120", "N-t70n11xx_120",
]

QAP_INSTANCES = [
    "tai15a", "tai15b", "nug17", "nug18", "nug20", "tai20a", "tai20b",
    "nug21", "tai25a", "tai25b", "bur26a", "bur26b", "bur26c", "bur26d",
    "tai30a", "tai30b", "tai35a", "tai35b", "tai40a", "tai40b", "tai50a",
    "tai50b", "tai60a", "tai60b", "tai64c", "tai80a", "tai80b", "tai100a",
    "tai100b", "tai150b",
]

PFSP_INSTANCES = [
    f"tai{jobs}_{mach}_{rep}"
    for jobs in (50, 100)
    for mach in (5, 10, 20)
    for rep in range(5)
]

TSP_INSTANCES = [
    "burma14", "ulysses16", "gr17", "ulysses22", "gr24", "fri26", "bays29",
    "dantzig42", "swiss42", "gr48", "hk48", "eil51", "berlin52", "st70",
    "eil76", "pr76", "gr96", "rat99", "kroA100", "kroC100", "eil101",
    "pr107", "pr124", "ch130", "pr136", "gr137", "pr144", "kroA150",
    "ch150", "pr152",
]


def _load_lop(path: Path, name: str) -> dict:
    B = parse_lolib(path)
    n = B.shape[0]
    return dict(name=f"LOP-{name}", ptype="LOP", n_vars=n,
                fitness=load_lolib_instance(B), better="max", unit="objective")


def _load_qap(path: Path, name: str) -> dict:
    H, D = parse_qaplib(path)
    n = H.shape[0]
    return dict(name=f"QAP-{name}", ptype="QAP", n_vars=n,
                fitness=load_qaplib_instance(H, D), better="min", unit="cost")


def _parse_taillard_pfsp_ceberio(filepath: Path) -> np.ndarray:
    text = Path(filepath).read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]

    header = None
    body_start = 0
    for i, ln in enumerate(lines):
        tokens = ln.split()
        if len(tokens) >= 2 and all(re.fullmatch(r"-?\d+", t) for t in tokens[:2]):
            header = tokens
            body_start = i + 1
            break
    if header is None:
        raise ValueError(f"Could not find a numeric header line in {filepath}")
    n_jobs, n_machines = int(header[0]), int(header[1])

    all_numbers: list[int] = []
    for ln in lines[body_start:]:
        tokens = ln.split()
        if not all(re.fullmatch(r"-?\d+", t) for t in tokens):
            continue
        all_numbers.extend(int(x) for x in tokens)
        if len(all_numbers) >= n_machines * n_jobs:
            break

    raw = np.array(all_numbers[: n_machines * n_jobs], dtype=np.int64)
    return raw.reshape(n_machines, n_jobs).T


def _load_pfsp(path: Path, name: str) -> dict:
    P = _parse_taillard_pfsp_ceberio(path)
    n = P.shape[0]
    return dict(name=f"PFSP-{name}", ptype="PFSP", n_vars=n,
                fitness=load_taillard_instance(P), better="min", unit="makespan")


def _load_tsp(path: Path, name: str) -> dict:
    coords = parse_tsplib(path)
    n = coords.shape[0]
    return dict(name=f"TSP-{name}", ptype="TSP", n_vars=n,
                fitness=create_tsp_from_coordinates(coords), better="min",
                unit="tour length")


_LOADERS = {
    "LOP": (LOP_INSTANCES, "LOP", "", _load_lop),
    "QAP": (QAP_INSTANCES, "QAP", ".qap", _load_qap),
    "PFSP": (PFSP_INSTANCES, "PFSP", ".fsp", _load_pfsp),
    "TSP": (TSP_INSTANCES, "TSP", ".tsp", _load_tsp),
}


def build_ceberio_problems(benchmark_dir: Path, max_instances: int) -> list:
    problems = []
    missing = []
    for ptype, (names, subdir, ext, loader) in _LOADERS.items():
        for name in names[:max_instances]:
            path = benchmark_dir / subdir / f"{name}{ext}"
            if not path.exists():
                missing.append(str(path))
                continue
            problems.append(loader(path, name))
    if missing:
        print(f"WARNING: {len(missing)} instance file(s) not found, skipped:", file=sys.stderr)
        for m in missing[:10]:
            print(f"  - {m}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more", file=sys.stderr)
    return problems


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark-dir", type=Path, required=True,
                    help="root directory with lop/, qap/, pfsp/, tsp/ subfolders")
    # reuse the same run/plot options as the synthetic-instance script
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--pop", type=int, default=100)
    p.add_argument("--gen", type=int, default=60)
    p.add_argument("--selection-ratio", type=float, default=0.5)
    p.add_argument("--laplace", type=float, default=0.01)
    p.add_argument("--max-instances", type=int, default=30,
                    help="max instances per problem type, 1..30 (default 30 = full Ceberio benchmark)")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--base-seed", type=int, default=100)
    p.add_argument("--cd-alpha", type=float, default=0.05)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--track-evolution", action="store_true",
                    help="also record and plot per-generation diversity/convergence "
                         "(one extra run per algorithm x instance; adds runtime)")
    p.add_argument("--evolution-dir", type=Path, default=Path("evolution"),
                    help="root output folder for evolution plots (default: evolution/)")
    p.add_argument("--evolution-mode", choices=["aggregate", "per-instance"],
                    default="aggregate",
                    help="'aggregate' (default): one diversity + one convergence plot "
                         "per PROBLEM TYPE, averaged over its instances. "
                         "'per-instance': also save one pair of plots per individual "
                         "instance -- only recommended for a handful of instances, "
                         "not the full 120-instance benchmark.")
    p.add_argument("--evolution-max-instances", type=int, default=5,
                    help="max instances per problem type used for evolution tracking "
                         "(independent of --max-instances, which controls the main "
                         "fitness comparison; evolution tracking runs one extra "
                         "full EDA run per algorithm x instance, so keep this small)")
    p.add_argument("--evolution-only", action="store_true",
                    help="skip the main fitness comparison entirely (no results/ output) "
                         "and only build evolution/ -- use this when results/ already "
                         "exists from a previous run and you just want the evolution plots")
    return p.parse_args()


def main():
    args = parse_args()
    algorithms = build_algorithms(args)

    if args.evolution_only:
        evo_problems = build_ceberio_problems(args.benchmark_dir, args.evolution_max_instances)
        if not evo_problems:
            print("No instance files found -- check --benchmark-dir layout.", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(evo_problems)} Ceberio-benchmark instances for evolution tracking only.")
        track_evolution(evo_problems, algorithms, args, root=args.evolution_dir)
        return

    problems = build_ceberio_problems(args.benchmark_dir, args.max_instances)
    if not problems:
        print("No instance files found -- check --benchmark-dir layout.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(problems)} Ceberio-benchmark instances.")

    results, labels = run_experiment(args, problems, algorithms)
    per_problem_tables(problems, results, labels)
    rank_table(problems, results, labels, args.k)
    library_stats(problems, results, labels, args.k)
    if not args.no_plot:
        boxplot(problems, results, labels, Path("results/compare_representations_ceberio.png"))
    critical_difference_diagrams(problems, results, labels, args.k, args.cd_alpha,
                                  Path("results/cd_ceberio"))

    if args.track_evolution:
        evo_problems = build_ceberio_problems(args.benchmark_dir, args.evolution_max_instances)
        track_evolution(evo_problems, algorithms, args, root=args.evolution_dir)


if __name__ == "__main__":
    main()