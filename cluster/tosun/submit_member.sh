#!/bin/bash
# Usage: bash cluster/tosun/submit_member.sh CONFIG MODEL MEMBER [NUM_SHARDS]
# Dataset must already be frozen/shared. Submits generation (GPU), then the scoring channels in parallel
# (ground truth on CPU; ArmoRM and, if configured, BARTScore on one GPU each), then packaging.
# Never submits router training.
# Smoke test first on new code or a new model:  llmr build-queries on a small config, NUM_SHARDS=1.
set -euo pipefail
source "$(dirname "$0")/config.sh"
llmr_activate
CFG="$1"; MODEL="$2"; MEMBER="$3"; NUM_SHARDS="${4:-1}"
[[ "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_SHARDS must be positive" >&2; exit 1; }
read -r GPUS CHANNELS < <(python - "$CFG" "$MODEL" <<'PY'
import sys
from llm_routing.config import load_benchmark_config
from llm_routing.pipeline.shared import channels
from llm_routing.pipeline.shared.dataset import load_dataset, model_pool
cfg = load_benchmark_config(sys.argv[1])
load_dataset(cfg)  # hash-check before reserving GPUs
print(int(model_pool(cfg)[sys.argv[2]].get("gpus", 1)), ",".join(channels(cfg)))
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
# Scoring channels are independent of each other: all start once generation finished.
SCORED=()
for CHANNEL in ${CHANNELS//,/ }; do
    GPU=()
    # ArmoRM keeps the whole 8B reward model on one GPU (~16 GB in bf16); BART-large needs ~2 GB.
    [ "$CHANNEL" = ground_truth ] || GPU=(--gres="$(llmr_gres 1)")
    ID=$(sbatch "${COMMON[@]}" "${GPU[@]}" --dependency="afterok:$GEN" "$JOB" "$CFG" "$MODEL" "$MEMBER" "$NUM_SHARDS" "$CHANNEL")
    SCORED+=("${ID%%;*}")
done
AFTER=$(IFS=:; echo "${SCORED[*]}")
PACK=$(sbatch "${COMMON[@]}" --dependency="afterok:$AFTER" "$JOB" "$CFG" "$MODEL" "$MEMBER" "$NUM_SHARDS" package)
PACK="${PACK%%;*}"
echo "Collection submitted: generation=$GEN ($GPUS GPU) scoring[$CHANNELS]=${SCORED[*]} package=$PACK"
