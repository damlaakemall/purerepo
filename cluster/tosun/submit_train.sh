#!/bin/bash
# Submit training runs with the site settings from config.sh:
#   bash cluster/tosun/submit_train.sh configs/experiments/X.yaml [num_seeds] [key=value overrides...]
set -euo pipefail
source "$(dirname "$0")/config.sh"
CFG="$1"; SEEDS="${2:-3}"; shift; [ $# -gt 0 ] && shift
mkdir -p logs
sbatch --partition="$LLMR_PARTITION" ${LLMR_ACCOUNT:+--account=$LLMR_ACCOUNT} ${LLMR_QOS:+--qos=$LLMR_QOS} \
  --gres="$LLMR_GPU_GRES" --array=0-$((SEEDS-1)) cluster/tosun/jobs/train.sbatch "$CFG" "$@"
