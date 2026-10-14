#!/bin/bash
# One stage for one member/model, sharded by SLURM array index. Submitted by submit_member.sh,
# which sets --gres/--mem per stage and model.
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
echo "[$(date)] $STAGE $MODEL shard ${SLURM_ARRAY_TASK_ID:-0}/$NUM_SHARDS on $(hostname)"
case "$STAGE" in
  generate) llmr_require_gpu; llmr generate "${ARGS[@]}" --max-retries 2 --batch-size "${LLMR_BATCH_SIZE:-256}" ;;
  armorm) llmr_require_gpu; llmr score "${ARGS[@]}" --scorer armorm ;;
  ground_truth) llmr score "${ARGS[@]}" --scorer ground_truth ;;   # CPU; MBPP runs model code in a subprocess
  package) llmr package-contribution "${ARGS[@]}" ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 1 ;;
esac
