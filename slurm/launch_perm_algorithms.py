"""
Generate sbatch commands for the permutation-EDA benchmark (one job per
combination of seed, problem, instance and algorithm).

Loops over:
    * seeds       1 .. 20
    * problems    LOP, QAP, PFSP, TSP
    * instances   every instance file found under <benchmark_dir>/{LOP,QAP,PFSP,TSP}/
    * algorithms  the 16 algorithms of scripts/compare_representations2.py
                  (9 bijective-coding + 7 permutation-distribution models)

Fixed run parameters: pop=400, gen=60, selection_ratio=0.5.

Full default grid = 4 problems x 30 instances x 16 algorithms x 20 seeds = 38400
jobs.  Keep <= 400 jobs running simultaneously, e.g. submit in slices or by
family::

    python3 slurm/launch_perm_algorithms.py | head -400 | bash
    python3 slurm/launch_perm_algorithms.py | sed -n '401,800p' | bash
    python3 slurm/launch_perm_algorithms.py | grep ' LOP '  | bash
    python3 slurm/launch_perm_algorithms.py | grep ' Mallows-K ' | bash

Run this from the submission directory (the one holding scripts/, slurm/,
Instances/, results/ and outputs/).  This launcher is self-contained (standard
library only), so plain ``python3`` runs it; the jobs themselves launch Python
through ``bnd -exec`` (see slurm/slurm_perm_algs.sh), which uses the project's
pipenv environment where perm_pateda is installed.

The benchmark directory (root holding LOP/ QAP/ PFSP/ TSP subfolders) defaults to
``Instances``; override with the environment variable ``PERM_BENCHMARK_DIR``.

Before the first submission, make sure outputs/ and results/ exist::

    mkdir -p outputs results
"""

import os
import sys
from pathlib import Path

# Fixed run parameters (as requested).
POP_SIZE = 1000
N_GEN = 250
SEL_RATIO = 0.5
SEEDS = range(6, 11)                     # 1 .. 20

SBATCH_SCRIPT = "slurm/slurm_perm_algs.sh"

# Resolve the project root (the dir holding slurm/ and Instances/) from this
# file's location, so the launcher works regardless of the current directory.
ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = os.environ.get("PERM_BENCHMARK_DIR", "Instances")

# The 16 algorithms of scripts/compare_representations.build_algorithms
# (9 bijective-coding models + 7 permutation-distribution models).
ALGORITHMS = [
    "Lehmer-UMDA", "Lehmer-Tree", "Lehmer-Markov",
    "FY-UMDA", "FY-Tree", "FY-Markov",
    "IV-UMDA", "IV-Tree", "IV-Markov",
    "Mallows-K", "Mallows-C", "GM-C", "PL", "EHM", "NHM", "DSM-AS",
]

# Problem type -> (subfolder, instance-file extension); mirrors the loaders in
# scripts/compare_representations2.py.
PROBLEM_LAYOUT = {
    "LOP": ("LOP", ""),
    "QAP": ("QAP", ".qap"),
    "PFSP": ("PFSP", ".fsp"),
    "TSP": ("TSP", ".tsp"),
}


def instances_for(problem: str) -> list:
    """Instance names (without extension) discovered on disk for a problem type."""
    subdir, ext = PROBLEM_LAYOUT[problem]
    folder = ROOT / BENCHMARK_DIR / subdir
    if not folder.is_dir():
        print(f"# missing folder: {folder}", file=sys.stderr)
        return []
    names = []
    for f in sorted(folder.iterdir()):
        if not f.is_file() or f.name.startswith("."):
            continue                     # skip hidden/junk files (e.g. .DS_Store)
        if ext:
            if f.name.endswith(ext):
                names.append(f.name[: -len(ext)])
        else:
            names.append(f.name)         # LOP files have no extension
    return names


if __name__ == "__main__":
    n_jobs = 0
    try:
        for problem in PROBLEM_LAYOUT:
            for instance in instances_for(problem):
                for alg in ALGORITHMS:
                    for seed in SEEDS:
                        print(f"sbatch {SBATCH_SCRIPT} {seed} {alg} {problem} "
                              f"{instance} {BENCHMARK_DIR} {POP_SIZE} {N_GEN} {SEL_RATIO}")
                        n_jobs += 1
    except BrokenPipeError:               # e.g. when piping through `head`
        pass
    else:
        print(f"# emitted {n_jobs} sbatch commands", file=sys.stderr)
