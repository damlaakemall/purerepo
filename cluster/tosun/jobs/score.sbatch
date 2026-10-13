#!/bin/bash
# Score one model's responses (BARTScore needs a GPU; exact-match is CPU). Array over models:
#   sbatch --array=0-4 cluster/tosun/jobs/score.sbatch configs/benchmarks/ours_v1.yaml
#SBATCH --job-name=llmr-score
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source cluster/tosun/config.sh
llmr_activate
CFG="$1"
MODEL=$(python -c "from llm_routing.config import load_benchmark_config as l; import sys; print(l('$CFG')['models'][int(sys.argv[1])]['name'])" "$SLURM_ARRAY_TASK_ID")
llmr score "$CFG" --model "$MODEL"
