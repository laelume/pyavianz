#!/usr/bin/env bash
# One-time setup on Rāpoi: create the virtualenv and install dependencies.
# Run inside an interactive compute session (not on the login node):
#   srun --partition=quicktest --cpus-per-task=4 --mem=8G --time=1:00:00 --pty bash
#   bash hpc/slurm/setup_env.sh
set -euo pipefail

export AVIANZ_HPC_SETUP=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

echo "Creating ${VENV} with $(python3 --version)"
mkdir -p "$(dirname "${VENV}")"
python3 -m venv "${VENV}"
source "${VENV}/bin/activate"
pip install --upgrade pip wheel
pip install -r "${HPC_DIR}/requirements-hpc.txt"

echo
echo "Checking imports..."
python - <<'EOF'
import torch, timm
import avianz_hpc.features, avianz_hpc.core.models
print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| timm", timm.__version__)
print("AviaNZ spectrogram import OK")
EOF

if [[ ! -s "${AVIANZ_HPC_PRETRAINED:-/nonexistent}" ]] || [[ $(stat -c %s "${AVIANZ_HPC_PRETRAINED}") -lt 1000000 ]]; then
    echo
    echo "NOTE: BirdCLEF weights not found at AVIANZ_HPC_PRETRAINED=${AVIANZ_HPC_PRETRAINED:-unset}."
    echo "      Copy the real model_fold0.pth (68 MB, from the model-testing Git LFS store) there,"
    echo "      or RegNet training starts from random weights."
fi
echo "Done."
