#!/usr/bin/env bash
# Shared environment for every avianz-hpc job. Source it; do not run it.
#   source "$HPC_DIR/slurm/env.sh"

# Inside a Slurm job $0 is a spooled copy, so the submit scripts pass HPC_DIR
# with --export. Fall back to this file's location for interactive use.
HPC_DIR="${HPC_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HPC_DIR
export AVIANZ_REPO="$(cd "${HPC_DIR}/.." && pwd)"

CONFIG="${AVIANZ_HPC_CONFIG:-${HPC_DIR}/config/project.env}"
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: ${CONFIG} not found. Copy hpc/config/project.env.example to project.env and edit it." >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck source=/dev/null
source "${CONFIG}"

if command -v module >/dev/null 2>&1 && [[ -n "${MODULES:-}" ]]; then
    module purge
    # shellcheck disable=SC2086
    module load ${MODULES}
fi

if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
elif [[ "${AVIANZ_HPC_SETUP:-0}" != "1" ]]; then
    echo "ERROR: virtualenv ${VENV} missing; run hpc/slurm/setup_env.sh first." >&2
    return 1 2>/dev/null || exit 1
fi

export PYTHONPATH="${HPC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
# Feature extraction parallelises over processes; keep each one single-threaded.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${WORK}/logs"
