#!/bin/bash
# One-time setup on Tosun (run on the login node):  bash cluster/tosun/setup_env.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"
llmr_load_modules
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$LLMR_CONDA_ENV" python=3.11
conda activate "$LLMR_CONDA_ENV"
cd "$LLMR_REPO"
pip install -e ".[gpu,tracking]"
pip install vllm || echo "vllm failed to install — set generation.engine=hf in the benchmark config"
mkdir -p "$LLMR_DATA_ROOT" logs
echo
echo "Done. Next (once per machine):"
echo "  wandb login                # for experiment tracking"
echo "  huggingface-cli login      # for gated models (Llama) and pushing/pulling benchmarks"
