#!/bin/bash
# Usage: bash cluster/tosun/submit_member.sh CONFIG MODEL MEMBER [NUM_SHARDS]
# Dataset must already be frozen/shared. This submits collection only, never router training.
set -euo pipefail
source "$(dirname "$0")/config.sh"
llmr_activate
CFG="$1"; MODEL="$2"; MEMBER="$3"; NUM_SHARDS="${4:-1}"
[[ "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_SHARDS must be positive" >&2; exit 1; }
ENGINE=$(python - "$CFG" "$MODEL" <<'PY'
import sys
from llm_routing.config import load_benchmark_config
from llm_routing.pipeline.shared.dataset import load_dataset, model_pool
from llm_routing.pipeline.shared.collect import generation_config
cfg = load_benchmark_config(sys.argv[1])
load_dataset(cfg)  # hash-check before reserving GPUs
print(generation_config(cfg, model_pool(cfg)[sys.argv[2]])['engine'])
PY
)
[[ "$LLMR_PARTITION" != TODO* ]] || { echo "Set Tosun partition in config.sh first" >&2; exit 1; }
mkdir -p logs
COMMON=(--parsable --partition="$LLMR_PARTITION" --array="0-$((NUM_SHARDS-1))")
[ -z "$LLMR_ACCOUNT" ] || COMMON+=(--account="$LLMR_ACCOUNT")
[ -z "$LLMR_QOS" ] || COMMON+=(--qos="$LLMR_QOS")
GPU=()
[ "$ENGINE" = api ] || GPU+=(--gres="$LLMR_GPU_GRES")
JOB=cluster/tosun/jobs/shared_member.sbatch
GEN=$(sbatch "${COMMON[@]}" "${GPU[@]}" "$JOB" "$CFG" "$MODEL" "$MEMBER" "$NUM_SHARDS" generate)
GEN="${GEN%%;*}"
GT=$(sbatch "${COMMON[@]}" --dependency="afterok:$GEN" "$JOB" "$CFG" "$MODEL" "$MEMBER" "$NUM_SHARDS" ground_truth)
GT="${GT%%;*}"
RM=$(sbatch "${COMMON[@]}" --gres="$LLMR_GPU_GRES" --dependency="afterok:$GT" "$JOB" "$CFG" "$MODEL" "$MEMBER" "$NUM_SHARDS" armorm)
RM="${RM%%;*}"
PACK=$(sbatch "${COMMON[@]}" --dependency="afterok:$RM" "$JOB" "$CFG" "$MODEL" "$MEMBER" "$NUM_SHARDS" package)
echo "Collection submitted: generation=$GEN GT=$GT ArmoRM=$RM package=$PACK"
