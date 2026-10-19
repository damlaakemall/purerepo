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
# torch and vLLM must match the driver. Tosun's is 570.x (CUDA 12.x); the PyPI wheels of vLLM >= 0.31 and its
# torch 2.13 are built for CUDA 13 and see no GPU there. vLLM's +cu129 release wheel with torch from the cu129
# index works on any 12.x driver (CUDA minor-version compatibility). Pinned so every member gets the same stack.
VLLM_VERSION="${VLLM_VERSION:-0.31.0}"; TORCH_CUDA="${TORCH_CUDA:-cu129}"
pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
pip install -e ".[gpu,graph,tracking,dev]"
pip install "https://github.com/vllm-project/vllm/releases/download/v$VLLM_VERSION/vllm-$VLLM_VERSION%2B$TORCH_CUDA-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --extra-index-url "https://download.pytorch.org/whl/$TORCH_CUDA" \
    || echo "vllm failed to install -- set generation.engine: hf for the affected models"
# vLLM imports torchcodec (audio inputs) at startup; its PyPI wheel links CUDA 13 and fails to load. Text-only
# generation needs no GPU decoding, so the CPU build of the same version is enough.
pip install --force-reinstall --no-deps "torchcodec==$(pip show torchcodec | awk '/^Version/{print $2}' | cut -d+ -f1)" \
    --index-url https://download.pytorch.org/whl/cpu
python -c "import vllm.multimodal"
python -c "import torch; assert torch.version.cuda.startswith('12'), f'torch built for CUDA {torch.version.cuda}; Tosun driver is 12.x'"
mkdir -p "$LLMR_DATA_ROOT" logs
echo
echo "Done. Next (once per machine):"
echo "  hf auth login              # model downloads and pushing/pulling releases"
echo "  wandb login                # experiment tracking (optional)"
echo "Then on a GPU node:  srun -p $LLMR_PARTITION --qos=$LLMR_QOS --gres=$(llmr_gres 1) --pty bash -c 'source cluster/tosun/config.sh; llmr_activate; llmr_require_gpu'"
