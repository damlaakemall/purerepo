#!/bin/bash
# Site settings for Sabancı University's Tosun cluster (SLURM). Every job script sources this file.
# Partition/QOS, modules and the excluded nodes are the ones SU Lab LLM-Routing-Project runs with on
# Tosun (scripts/slurm/run_generate.sh there). Override any of them from the environment.
# `sinfo -p cuda -o "%20N %12G %8m %t"` lists the GPUs and memory of each node.

export LLMR_REPO="${LLMR_REPO:-$HOME/purerepo}"                 # where the repo is cloned (/cta/users/<you>/purerepo)
export LLMR_DATA_ROOT="${LLMR_DATA_ROOT:-$HOME/llmr_data}"       # releases and contributions (large)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"           # model downloads: ~300 GB for the 12-model pool
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export LLMR_VENV="${LLMR_VENV:-$LLMR_REPO/.venv}"

# SLURM defaults used by jobs/*.sbatch (can be overridden on the sbatch command line)
export LLMR_PARTITION="${LLMR_PARTITION:-cuda}"
export LLMR_QOS="${LLMR_QOS:-cuda}"
export LLMR_ACCOUNT="${LLMR_ACCOUNT:-}"
export LLMR_GPU_TYPE="${LLMR_GPU_TYPE:-}"                       # e.g. a100 -> --gres=gpu:a100:N; empty = any GPU
# K80 (cn05,cn06; also excluded by the reference repo) and V100 (cn04,cn11,gpu01) cards have no bf16, which every
# pool model and ArmoRM load in. Left: RTX PRO 6000 96 GB, A100 80 GB, H200, A30/A5000/RTX 4500 24 GB, A4000 16 GB.
export LLMR_EXCLUDE="${LLMR_EXCLUDE:-cn04,cn05,cn06,cn11,gpu01}"

# --gres string for N GPUs: llmr_gres 2 -> gpu:2 (or gpu:a100:2)
llmr_gres() {
  echo "gpu:${LLMR_GPU_TYPE:+$LLMR_GPU_TYPE:}${1:-1}"
}

llmr_load_modules() {
  module purge 2>/dev/null || true
  module load python/3.12.9
  module load cuda/12.8 2>/dev/null || true   # refuses to load on the (GPU-less) login node; jobs get it
  # python/3.12.9's _sqlite3 is linked (RPATH) to a 2020 libsqlite3 without sqlite3_deserialize, so
  # `import sqlite3` fails, and vLLM imports it. Preload the sqlite module's newer library instead.
  module load sqlite/3.46.0-36f2sjv
  export LD_PRELOAD="$(dirname "$(command -v sqlite3)")/../lib/libsqlite3.so${LD_PRELOAD:+:$LD_PRELOAD}"
}

llmr_activate() {
  llmr_load_modules
  # shellcheck disable=SC1091
  source "$LLMR_VENV/bin/activate"
  cd "$LLMR_REPO" || exit 1
}

# A torch built for a newer CUDA than the driver sees no GPU and silently runs on CPU (~750x slower);
# the job then dies on the wall clock hours later. GPU jobs call this first and stop instead.
llmr_require_gpu() {
  python - <<'PYCHECK'
import sys, torch
if not torch.cuda.is_available():
    sys.exit(f"ERROR: torch {torch.__version__} (CUDA {torch.version.cuda}) sees no GPU; reinstall torch for the "
             "driver shown by nvidia-smi, e.g. pip install torch --index-url https://download.pytorch.org/whl/cu128")
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) -> {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
PYCHECK
}
