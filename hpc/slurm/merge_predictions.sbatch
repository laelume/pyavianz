#!/usr/bin/env bash
#SBATCH --job-name=avz-merge-pred
#SBATCH --partition=quicktest
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
# afterany (not afterok) dependency: summarise whatever finished, even if a
# shard failed, so it can be resubmitted afterwards.
# Usage: sbatch merge_predictions.sbatch OUT_DIR [--windows]
set -euo pipefail
source "${HPC_DIR}/slurm/env.sh"

python -m avianz_hpc merge-predictions "$@"
