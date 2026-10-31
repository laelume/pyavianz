#!/usr/bin/env bash
#SBATCH --job-name=solar-survey
#SBATCH --partition=parallel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10-00:00:00
#SBATCH --output=%x_%j.out
# Runs an AviaNZ recogniser over a list of SoLAR folders (see solar_survey.py).
# One job processes the folders one after another; the Rāpoi docs ask for few
# concurrent smbclient sessions. Resubmit the same command to continue.
#
# Usage:  sbatch ~/avianz/hpc/solar/solar_survey.sbatch ~/folders.txt [solar_survey.py run options]
# Needs:  ~/avianz_env.sh (modules + venv + PYTHONPATH) and an smbclient
#         authentication file at $SOLAR_AUTH_FILE (default ~/.solar_auth, chmod 600).
set -euo pipefail

source ~/avianz_env.sh
export SOLAR_AUTH_FILE="${SOLAR_AUTH_FILE:-$HOME/.solar_auth}"

STAGE_BASE="/tmp/${USER}/${SLURM_JOB_ID}"
trap 'rm -rf "${STAGE_BASE}"' EXIT

FOLDERS="$1"; shift
python ~/avianz/hpc/solar/solar_survey.py run "${FOLDERS}" "$@"
