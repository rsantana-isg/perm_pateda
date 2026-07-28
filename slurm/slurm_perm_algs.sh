#!/bin/bash
#
# SLURM template for ONE permutation-EDA experiment
# (one seed x one algorithm x one benchmark instance).
#
# Positional arguments mirror scripts/run_perm_experiment.py (no flags, seed first):
#   $1 = seed             random seed
#   $2 = alg              algorithm label (Mallows-K, Lehmer-UMDA, EHM, ...)
#   $3 = problem          LOP | QAP | PFSP | TSP
#   $4 = instance         instance name without extension (e.g. N-be75eec, tai15a,
#                         tai50_5_0, burma14)
#   $5 = benchmark_dir    root directory with LOP/ QAP/ PFSP/ TSP subfolders
#   $6 = pop_size
#   $7 = n_gen
#   $8 = selection_ratio
#
# Output:
#   A self-describing .dat file written to results/ whose name encodes every
#   parameter.  It contains the best and mean fitness at every generation (natural
#   objective units), the population diversity per generation, the final best
#   solution with its fitness, and the wall-clock time.  Files are skipped if they
#   already exist, so re-launching is idempotent.

#SBATCH --job-name=perm_eda
#SBATCH --output=outputs/PERM_%A_%a.out
#SBATCH --error=outputs/PERM_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=CPU

set -euo pipefail

if [ "$#" -ne 8 ]; then
    echo "Usage: sbatch slurm_perm_algs.sh seed alg problem instance benchmark_dir pop gen sel_ratio" >&2
    exit 1
fi

SEED="$1"
ALG="$2"
PROBLEM="$3"
INSTANCE="$4"
BENCHMARK_DIR="$5"
POP_SIZE="$6"
N_GEN="$7"
SEL_RATIO="$8"

# Resolve the repository root so the script works regardless of where SLURM
# spawned us from.  This file lives in slurm/ alongside launch_perm_algorithms.py.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p results outputs

# Self-describing output filename (all parameters encoded).
OUTPUT_FILE="results/perm_${PROBLEM}_${ALG}_${INSTANCE}_pop${POP_SIZE}_gen${N_GEN}_sel${SEL_RATIO}_seed${SEED}.dat"

# Skip if already done (idempotent re-launches).
if [ -f "${OUTPUT_FILE}" ]; then
    echo "Skipping: ${OUTPUT_FILE} already exists."
    exit 0
fi

# Interpreter where perm_pateda / pateda are installed.  Override with PERM_PYTHON.
PYTHON_BIN="${PERM_PYTHON:-python3}"

echo "Running: ${PYTHON_BIN} scripts/run_perm_experiment.py ${SEED} ${ALG} ${PROBLEM} ${INSTANCE} ${BENCHMARK_DIR} ${POP_SIZE} ${N_GEN} ${SEL_RATIO}"

# Write to a temporary file first, then move it into place, so an interrupted job
# never leaves a truncated .dat that the skip-check would treat as complete.
TMP_FILE="$(mktemp "${OUTPUT_FILE}.XXXXXX")"
"${PYTHON_BIN}" scripts/run_perm_experiment.py \
    "${SEED}" "${ALG}" "${PROBLEM}" "${INSTANCE}" "${BENCHMARK_DIR}" \
    "${POP_SIZE}" "${N_GEN}" "${SEL_RATIO}" \
    > "${TMP_FILE}"
mv -f "${TMP_FILE}" "${OUTPUT_FILE}"

# Example usage:
# sbatch slurm/slurm_perm_algs.sh 1 Mallows-K LOP N-be75eec Instances 400 60 0.5
