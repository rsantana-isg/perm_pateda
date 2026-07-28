#!/usr/bin/env python3
"""Generate sbatch commands for the permutation-EDA benchmark on a cluster.

Prints one ``sbatch slurm/slurm_perm_algs.sh ...`` line per
``(problem, instance, algorithm, seed)`` combination, looping over:

    * seeds       1 .. 20
    * problems    LOP, QAP, PFSP, TSP
    * instances   every instance of each problem found under the benchmark dir
                  (the same 30-per-type lists used by
                  scripts/compare_representations2.py)
    * algorithms  all algorithms of scripts/compare_representations2.py
                  (= compare_representations.build_algorithms: 9 bijective-coding
                  + 7 permutation-distribution models = 16 algorithms)

Fixed run parameters: pop=400, gen=60, selection_ratio=0.5.

Full grid = 4 problems x 30 instances x 16 algorithms x 20 seeds = 38400 jobs.
The launcher is NON-DESTRUCTIVE: it only prints commands.  Keep <= 400 jobs
running simultaneously, e.g. submit in slices or by family::

    python3 slurm/launch_perm_algorithms.py | head -400 | bash
    python3 slurm/launch_perm_algorithms.py | sed -n '401,800p' | bash
    python3 slurm/launch_perm_algorithms.py | grep ' LOP '  | bash
    python3 slurm/launch_perm_algorithms.py | grep ' Mallows-K ' | bash

The benchmark directory (root holding LOP/ QAP/ PFSP/ TSP subfolders) defaults to
``Instances`` (relative to the repository root); override with the environment
variable ``PERM_BENCHMARK_DIR``.  Instances whose files are absent are skipped
(reported on stderr), so re-running after adding files only emits the new jobs.

Before the FIRST submission, create the SLURM log directory (its
``#SBATCH --output=outputs/...`` path is resolved by SLURM before the job body's
own ``mkdir`` runs)::

    mkdir -p outputs results

After all jobs finish, harvest results/perm_*.dat for analysis.
"""
import os
import sys
from pathlib import Path

# Make the sibling scripts importable to reuse the exact instance lists and the
# algorithm registry, so the cluster grid matches the local comparison scripts.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import argparse  # noqa: E402
from compare_representations2 import (  # noqa: E402
    LOP_INSTANCES, QAP_INSTANCES, PFSP_INSTANCES, TSP_INSTANCES, _LOADERS,
)
from compare_representations import build_algorithms  # noqa: E402


# Fixed run parameters (as requested).
POP_SIZE = 400
N_GEN = 60
SEL_RATIO = 0.5
SEEDS = range(1, 21)                       # 1 .. 20

SBATCH_SCRIPT = "slurm/slurm_perm_algs.sh"
BENCHMARK_DIR = os.environ.get("PERM_BENCHMARK_DIR", "Instances")

PROBLEMS = {
    "LOP": LOP_INSTANCES,
    "QAP": QAP_INSTANCES,
    "PFSP": PFSP_INSTANCES,
    "TSP": TSP_INSTANCES,
}

ALGORITHMS = [label for label, _cls, _extra in
              build_algorithms(argparse.Namespace(laplace=0.01))]


def _instance_path(problem: str, instance: str) -> Path:
    _names, subdir, ext, _loader = _LOADERS[problem]
    return _REPO_ROOT / BENCHMARK_DIR / subdir / f"{instance}{ext}"


def main() -> None:
    n_jobs = 0
    n_missing = 0
    try:
        for problem, instances in PROBLEMS.items():
            for instance in instances:
                if not _instance_path(problem, instance).exists():
                    n_missing += 1
                    print(f"# missing: {_instance_path(problem, instance)}", file=sys.stderr)
                    continue
                for alg in ALGORITHMS:
                    for seed in SEEDS:
                        print(f"sbatch {SBATCH_SCRIPT} {seed} {alg} {problem} "
                              f"{instance} {BENCHMARK_DIR} {POP_SIZE} {N_GEN} {SEL_RATIO}")
                        n_jobs += 1
    except BrokenPipeError:            # e.g. when piping through `head`
        return
    print(f"# emitted {n_jobs} sbatch commands "
          f"({n_missing} instance(s) skipped as missing)", file=sys.stderr)


if __name__ == "__main__":
    main()
