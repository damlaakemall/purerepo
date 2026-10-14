#!/bin/bash
# One-time setup on Tosun (run on the login node):  bash cluster/tosun/setup_env.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"
llmr_load_modules
cd "$LLMR_REPO"
[ -d "$LLMR_VENV" ] || python -m venv "$LLMR_VENV"
# shellcheck disable=SC1091
source "$LLMR_VENV/bin/activate"
pip install --upgrade pip
# torch must match the driver (570.x = CUDA 12.8); the default PyPI wheel may target a newer CUDA.
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[gpu,tracking,dev]"
pip install vllm || echo "vllm failed to install -- set generation.engine: hf for the affected models"
mkdir -p "$LLMR_DATA_ROOT" logs
echo
echo "Done. Next (once per machine):"
echo "  hf auth login              # model downloads and pushing/pulling releases"
echo "  wandb login                # experiment tracking (optional)"
echo "Then on a GPU node:  srun -p $LLMR_PARTITION --qos=$LLMR_QOS --gres=$(llmr_gres 1) --pty bash -c 'source cluster/tosun/config.sh; llmr_activate; llmr_require_gpu'"
