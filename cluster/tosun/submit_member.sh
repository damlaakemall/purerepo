#!/bin/bash
# Usage: bash cluster/tosun/submit_member.sh CONFIG MODEL MEMBER [NUM_SHARDS]
# Dataset must already be frozen/shared. Submits generation (GPU), then ground-truth (CPU) and ArmoRM
# (GPU) scoring in parallel, then packaging. Never submits router training.
# Smoke test first on new code or a new model:  llmr build-queries on a small config, NUM_SHARDS=1.
set -euo pipefail
source "$(dirname "$0")/config.sh"
llmr_activate
CFG="$1"; MODEL="$2"; MEMBER="$3"; NUM_SHARDS="${4:-1}"
[[ "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_SHARDS must be positive" >&2; exit 1; }
GPUS=$(python - "$CFG" "$MODEL" <<'PY'
import sys
from llm_routing.config import load_benchmark_config
from llm_routing.pipeline.shared.dataset import load_dataset, model_pool
cfg = load_benchmark_config(sys.argv[1])
load_dataset(cfg)  # hash-check before reserving GPUs
print(int(model_pool(cfg)[sys.argv[2]].get("gpus", 1)))
PY
)
mkdir -p logs
COMMON=(--parsable --partition="$LLMR_PARTITION" --array="0-$((NUM_SHARDS-1))")
[ -z "$LLMR_ACCOUNT" ] || COMMON+=(--account="$LLMR_ACCOUNT")
[ -z "$LLMR_QOS" ] || COMMON+=(--qos="$LLMR_QOS")
[ -z "$LLMR_EXCLUDE" ] || COMMON+=(--exclude="$LLMR_EXCLUDE")
JOB=cluster/tosun/jobs/shared_member.sbatch
GEN=$(sbatch "${COMMON[@]}" --gres="$(llmr_gres "$GPUS")" --mem=$((64 * GPUS))G --cpus-per-task=$((8 * GPUS)) \
      "$JOB" "$CFG" "$MODEL" "$MEMBER" "$NUM_SHARDS" generate)
GEN="${GEN%%;*}"
# Scoring channels are independent of each other: both start once generation finished.
GT=$(sbatch "${COMMON[@]}" --dependency="afterok:$GEN" "$JOB" "$CFG" "$MODEL" "$MEMBER" "$NUM_SHARDS" ground_truth)
GT="${GT%%;*}"
# ArmoRM keeps the whole 8B reward model on one GPU (~16 GB in bf16).
RM=$(sbatch "${COMMON[@]}" --gres="$(llmr_gres 1)" --dependency="afterok:$GEN" "$JOB" "$CFG" "$MODEL" "$MEMBER" "$NUM_SHARDS" armorm)
RM="${RM%%;*}"
PACK=$(sbatch "${COMMON[@]}" --dependency="afterok:$GT:$RM" "$JOB" "$CFG" "$MODEL" "$MEMBER" "$NUM_SHARDS" package)
echo "Collection submitted: generation=$GEN ($GPUS GPU) GT=$GT ArmoRM=$RM package=$PACK"
