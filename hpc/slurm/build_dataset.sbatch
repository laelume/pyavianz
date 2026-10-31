#!/usr/bin/env bash
#SBATCH --job-name=avz-build
#SBATCH --partition=parallel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%A_%a.out
# Array job, one shard per task. Submit with --array=0-$((NUM_SHARDS-1)) and
# --export=ALL,HPC_DIR=...,NUM_SHARDS=N (submit_train.sh does this).
# Usage: sbatch --array=0-31 build_dataset.sbatch MANIFEST DATASET_DIR [build-dataset args...]
set -euo pipefail
source "${HPC_DIR}/slurm/env.sh"

MANIFEST="$1"; DATASET="$2"; shift 2
python -m avianz_hpc build-dataset "${MANIFEST}" "${DATASET}" \
    --shard "${SLURM_ARRAY_TASK_ID}" --num-shards "${NUM_SHARDS}" "$@"
