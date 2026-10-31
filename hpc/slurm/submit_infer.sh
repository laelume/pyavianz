#!/usr/bin/env bash
# Submit manifest -> infer (array) -> merge-predictions. Run on the login node.
#
# Usage: hpc/slurm/submit_infer.sh RUN_NAME [infer args...]
#   infer args must include --classifier and/or --detector (model directories)
# Examples:
#   hpc/slurm/submit_infer.sh survey2025 --classifier "$WORK/models/kiwi_ruru_cls"
#   hpc/slurm/submit_infer.sh survey2025 --detector "$WORK/models/bird_det" \
#       --classifier "$WORK/models/kiwi_ruru_cls" --detector-threshold 0.3
# Env overrides:
#   INFER_DEVICE=cpu     run on CPU_PARTITION without GPUs
#   MAX_PARALLEL=4       how many array tasks run at once (default 4; Rāpoi has 6 A100s)
#   MANIFEST=path.tsv    reuse an existing manifest instead of scanning INFER_AUDIO_ROOTS
set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HPC_DIR AVIANZ_HPC_SETUP=1
source "${HPC_DIR}/config/project.env"

RUN="$1"; shift
OUT="${WORK}/inference/${RUN}"
LOGS="${WORK}/logs"
mkdir -p "${LOGS}" "${OUT}" "${WORK}/manifests"

# shellcheck disable=SC2086
SB=(sbatch --parsable --export=ALL,HPC_DIR="${HPC_DIR}" ${SBATCH_EXTRA})
DEP=()
if [[ -z "${MANIFEST:-}" ]]; then
    MANIFEST="${WORK}/manifests/${RUN}_infer.tsv"
    # shellcheck disable=SC2086
    J1=$("${SB[@]}" --output="${LOGS}/%x_%j.out" "${HPC_DIR}/slurm/manifest.sbatch" \
        "${MANIFEST}" ${INFER_AUDIO_ROOTS})
    DEP=(--dependency=afterok:${J1})
    echo "manifest ${J1} -> ${MANIFEST}"
fi

if [[ "${INFER_DEVICE:-gpu}" == "cpu" ]]; then
    RES=(--partition="${CPU_PARTITION}")
    DEVICE_ARGS=(--device cpu)
else
    RES=(--partition="${GPU_PARTITION}" --gres=gpu:1)
    DEVICE_ARGS=()
fi

J2=$("${SB[@]}" "${DEP[@]}" "${RES[@]}" --array=0-$((INFER_SHARDS - 1))%${MAX_PARALLEL:-4} \
    --export=ALL,HPC_DIR="${HPC_DIR}",NUM_SHARDS="${INFER_SHARDS}" \
    --output="${LOGS}/%x_%A_%a.out" "${HPC_DIR}/slurm/infer.sbatch" \
    "${MANIFEST}" "${OUT}" "${DEVICE_ARGS[@]}" "$@")
J3=$("${SB[@]}" --dependency=afterany:${J2} --output="${LOGS}/%x_%j.out" \
    "${HPC_DIR}/slurm/merge_predictions.sbatch" "${OUT}")
echo "infer ${J2} (array of ${INFER_SHARDS}) -> merge ${J3}"
echo "Results: ${OUT}/detections.csv, ${OUT}/summary.csv, ${OUT}/annotations/"
