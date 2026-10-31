#!/usr/bin/env bash
#SBATCH --job-name=avz-merge-data
#SBATCH --partition=quicktest
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out
# Usage: sbatch merge_dataset.sbatch DATASET_DIR [merge-dataset args...]
set -euo pipefail
source "${HPC_DIR}/slurm/env.sh"

python -m avianz_hpc merge-dataset "$@"
