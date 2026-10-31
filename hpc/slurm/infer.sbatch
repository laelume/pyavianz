#!/usr/bin/env bash
#SBATCH --job-name=avz-infer
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A_%a.out
# Array job, one shard per task. Partition and GPU flags are given at submit
# time (submit_infer.sh) so the same script runs on gpu or CPU-only partitions.
# Spectrogram extraction is CPU-bound: the GPU is fed by cpus-per-task - 1
# worker processes, so raise --cpus-per-task before adding more GPUs.
# Re-submitting skips recordings that already have output.
# Usage: sbatch --array=0-7 --partition=gpu --gres=gpu:1 \
#            --export=ALL,HPC_DIR=...,NUM_SHARDS=8 infer.sbatch MANIFEST OUT_DIR [infer args...]
set -euo pipefail
source "${HPC_DIR}/slurm/env.sh"

MANIFEST="$1"; OUT="$2"; shift 2
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader || true
python -m avianz_hpc infer "${MANIFEST}" "${OUT}" \
    --shard "${SLURM_ARRAY_TASK_ID}" --num-shards "${NUM_SHARDS}" "$@"
