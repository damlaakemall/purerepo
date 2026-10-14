#!/bin/bash
# One stage for one member/model, sharded by SLURM array index.
#SBATCH --job-name=llmr-member
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source cluster/tosun/config.sh
llmr_activate
CFG="$1"; MODEL="$2"; MEMBER="$3"; NUM_SHARDS="$4"; STAGE="$5"
ARGS=("$CFG" --model "$MODEL" --member "$MEMBER" --num-shards "$NUM_SHARDS" --shard-index "${SLURM_ARRAY_TASK_ID:-0}")
case "$STAGE" in
  generate) llmr generate "${ARGS[@]}" --max-retries 2 ;;
  ground_truth|armorm) llmr score "${ARGS[@]}" --scorer "$STAGE" ;;
  package) llmr package-contribution "${ARGS[@]}" ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 1 ;;
esac
