#!/bin/bash
# Train + evaluate one experiment. Extra args are config overrides. Use an array for seeds:
#   sbatch --array=0-2 cluster/tosun/jobs/train.sbatch configs/experiments/hybrid_llm_deberta__ours_v1.yaml
#SBATCH --job-name=llmr-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source cluster/tosun/config.sh
llmr_activate
CFG="$1"; shift
SEED="${SLURM_ARRAY_TASK_ID:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"   # set to offline if compute nodes have no internet, then `wandb sync`
llmr train "$CFG" seed="$SEED" output_dir="$LLMR_REPO/runs" "$@"
