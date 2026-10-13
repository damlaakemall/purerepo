#!/bin/bash
# Generate k responses for ONE model. Submit as an array, one task per model:
#   sbatch --array=0-4 cluster/tosun/jobs/generate.sbatch configs/benchmarks/ours_v1.yaml
# Resumable: resubmitting continues where a killed job stopped.
#SBATCH --job-name=llmr-gen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source cluster/tosun/config.sh
llmr_activate
CFG="$1"
MODEL=$(python -c "from llm_routing.config import load_benchmark_config as l; import sys; print(l('$CFG')['models'][int(sys.argv[1])]['name'])" "$SLURM_ARRAY_TASK_ID")
echo "[$(date)] generating $MODEL on $(hostname)"; nvidia-smi -L || true
llmr generate "$CFG" --model "$MODEL"
