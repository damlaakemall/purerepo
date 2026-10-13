#!/bin/bash
# Site settings for Sabancı University's Tosun cluster (SLURM).
# Fill in the TODOs once: run `sinfo -o "%P %G %l %D"` on Tosun to see partitions, GPUs and time limits,
# and `module avail` for available modules. Then commit this file so the whole team uses the same settings.

export LLMR_REPO="${LLMR_REPO:-$HOME/purerepo}"                 # where the repo is cloned (/cta/users/<you>/purerepo)
export LLMR_DATA_ROOT="${LLMR_DATA_ROOT:-$HOME/llmr_data}"       # benchmarks (large); TODO: use scratch/shared storage if Tosun has it
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"           # model downloads (tens of GB) TODO: move to scratch if home quota is small
export LLMR_CONDA_ENV="${LLMR_CONDA_ENV:-llmr}"

# SLURM defaults used by jobs/*.sbatch (can be overridden on the sbatch command line)
export LLMR_PARTITION="${LLMR_PARTITION:-TODO_gpu_partition}"   # TODO: from `sinfo`
export LLMR_ACCOUNT="${LLMR_ACCOUNT:-}"                         # TODO: if Tosun requires --account / --qos
export LLMR_QOS="${LLMR_QOS:-}"
export LLMR_GPU_GRES="${LLMR_GPU_GRES:-gpu:1}"                  # e.g. gpu:a100:1 — TODO: from `sinfo -o "%G"`

# Modules to load before activating conda. TODO: adjust to `module avail` output.
llmr_load_modules() {
  module purge 2>/dev/null || true
  module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
  module load cuda 2>/dev/null || true
}

llmr_activate() {
  llmr_load_modules
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$LLMR_CONDA_ENV"
  cd "$LLMR_REPO" || exit 1
}
