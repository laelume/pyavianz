#!/usr/bin/env bash
#SBATCH --job-name=avz-manifest
#SBATCH --partition=quicktest
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
# Usage: sbatch manifest.sbatch OUTPUT.tsv ROOT [ROOT ...] [--annotated-only]
set -euo pipefail
source "${HPC_DIR}/slurm/env.sh"

python -m avianz_hpc manifest -o "$@"
