#!/usr/bin/env bash
#SBATCH --job-name=avz-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:30:00
#SBATCH --output=%x_%j.out
# The gpu partition allows 24 h. Training reads many small .npy files, so the
# dataset is staged to node-local /tmp (NVMe) first unless STAGE_TO_TMP=0.
# Usage: sbatch train.sbatch DATASET_DIR MODEL_OUT_DIR [train args...]
#   DATASET_DIR is a merge-dataset output; trains on its train/ and tests on test/.
set -euo pipefail
source "${HPC_DIR}/slurm/env.sh"

DATASET="$1"; OUT="$2"; shift 2
TRAIN_DIR="${DATASET}/train"
TEST_ARGS=()
[[ -f "${DATASET}/test/labels.json" ]] && TEST_ARGS=(--test-folder "${DATASET}/test")

if [[ "${STAGE_TO_TMP:-1}" == "1" ]]; then
    LOCAL="/tmp/${USER}/${SLURM_JOB_ID}"
    trap 'rm -rf "${LOCAL}"' EXIT
    mkdir -p "${LOCAL}"
    echo "Staging ${DATASET} -> ${LOCAL}"
    rsync -a --exclude shards "${DATASET}/" "${LOCAL}/"
    TRAIN_DIR="${LOCAL}/train"
    [[ ${#TEST_ARGS[@]} -gt 0 ]] && TEST_ARGS=(--test-folder "${LOCAL}/test")
fi

nvidia-smi --query-gpu=name,memory.total --format=csv
python -m avianz_hpc train "${TRAIN_DIR}" "${OUT}" "${TEST_ARGS[@]}" "$@"
