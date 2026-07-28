#!/bin/bash

# Job name
#SBATCH --job-name=perm_eda
# Define output and error files
#SBATCH --output=outputs/PERM_%A_%a.out
#SBATCH --error=outputs/PERM_%A_%a.err

# Resource requirements
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=CPU

# Parameters (seed first, matching scripts/run_perm_experiment.py):
# $1 = seed
# $2 = alg              (Lehmer-UMDA, Mallows-K, EHM, ...)
# $3 = problem          (LOP | QAP | PFSP | TSP)
# $4 = instance         (name without extension, e.g. N-t59b11xx, tai15a, tai50_5_0, burma14)
# $5 = benchmark_dir    (root with LOP/ QAP/ PFSP/ TSP subfolders, e.g. Instances)
# $6 = pop_size
# $7 = n_gen
# $8 = selection_ratio
#
# The job runs in the submission directory (which holds scripts/, Instances/,
# results/ and outputs/); Python is launched through the cluster's `bnd -exec`
# wrapper -- exactly like slurm/slurm_pbo.sh -- so it uses the project's pipenv
# environment where perm_pateda is installed.

mkdir -p results outputs

# Output filename encodes all parameters (self-describing results).
OUTPUT_FILE="results/perm_${3}_${2}_${4}_pop${6}_gen${7}_sel${8}_seed${1}.dat"

# Skip if already done (idempotent, safe to re-launch).
if [ -f "$OUTPUT_FILE" ]; then
    echo "Skipping: $OUTPUT_FILE already exists."
    exit 0
fi

echo "Executing: bnd -exec python3 scripts/run_perm_experiment.py $1 $2 $3 $4 $5 $6 $7 $8 > $OUTPUT_FILE"

# Execute the experiment (bnd -exec runs python inside the project environment).
bnd -exec python3 scripts/run_perm_experiment.py $1 $2 $3 $4 $5 $6 $7 $8 > "$OUTPUT_FILE"

# Example usage:
# sbatch slurm/slurm_perm_algs.sh 1 Lehmer-UMDA LOP N-t59b11xx Instances 400 60 0.5
