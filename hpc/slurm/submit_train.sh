#!/usr/bin/env bash
# Submit manifest -> build-dataset (array) -> merge-dataset -> train as one
# dependency chain. Run on the Rāpoi login node; it only calls sbatch.
#
# Usage: hpc/slurm/submit_train.sh NAME TASK [build-dataset args...] [-- train args...]
#   NAME   label for this dataset/model, e.g. morepork_v1
#   TASK   classifier | detector
# Examples:
#   hpc/slurm/submit_train.sh kiwi_ruru_cls classifier --species "Morepork,Kiwi (Nth Is Brown)"
#   hpc/slurm/submit_train.sh bird_det detector --background-fraction 0.5 -- --epochs 30 --patience 10
# Env overrides: SKIP_BUILD=1 (dataset exists), MERGE_ARGS="--max-per-class 5000 --group-by dir"
set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HPC_DIR AVIANZ_HPC_SETUP=1
source "${HPC_DIR}/config/project.env"

NAME="$1"; TASK="$2"; shift 2
BUILD_ARGS=(); TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then shift; TRAIN_ARGS=("$@"); break; fi
    BUILD_ARGS+=("$1"); shift
done

DATASET="${WORK}/datasets/${NAME}"
MODEL="${WORK}/models/${NAME}"
MANIFEST="${WORK}/manifests/${NAME}_train.tsv"
LOGS="${WORK}/logs"
mkdir -p "${LOGS}" "$(dirname "${MANIFEST}")" "${DATASET}" "${MODEL}"
[[ -n "${ANNOTATION_ROOT}" ]] && BUILD_ARGS=(--annotation-root "${ANNOTATION_ROOT}" "${BUILD_ARGS[@]}")

# shellcheck disable=SC2086
SB=(sbatch --parsable --export=ALL,HPC_DIR="${HPC_DIR}" ${SBATCH_EXTRA})
DEP=()
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    # shellcheck disable=SC2086
    J1=$("${SB[@]}" --output="${LOGS}/%x_%j.out" "${HPC_DIR}/slurm/manifest.sbatch" \
        "${MANIFEST}" ${TRAIN_AUDIO_ROOTS})
    J2=$("${SB[@]}" --dependency=afterok:${J1} --partition="${CPU_PARTITION}" \
        --array=0-$((BUILD_SHARDS - 1)) --export=ALL,HPC_DIR="${HPC_DIR}",NUM_SHARDS="${BUILD_SHARDS}" \
        --output="${LOGS}/%x_%A_%a.out" "${HPC_DIR}/slurm/build_dataset.sbatch" \
        "${MANIFEST}" "${DATASET}" --task "${TASK}" "${BUILD_ARGS[@]}")
    # shellcheck disable=SC2086
    J3=$("${SB[@]}" --dependency=afterok:${J2} --output="${LOGS}/%x_%j.out" \
        "${HPC_DIR}/slurm/merge_dataset.sbatch" "${DATASET}" ${MERGE_ARGS:-})
    DEP=(--dependency=afterok:${J3})
    echo "manifest ${J1} -> build ${J2} (array of ${BUILD_SHARDS}) -> merge ${J3}"
fi
J4=$("${SB[@]}" "${DEP[@]}" --partition="${GPU_PARTITION}" --output="${LOGS}/%x_%j.out" \
    "${HPC_DIR}/slurm/train.sbatch" "${DATASET}" "${MODEL}" "${TRAIN_ARGS[@]}")
echo "train ${J4} -> ${MODEL}"
echo "Logs: ${LOGS}    Watch: squeue -u ${USER}"
