#!/bin/bash
# Build OUR benchmark end to end with SLURM dependencies:
#   bash cluster/tosun/submit_pipeline.sh configs/benchmarks/ours_v1.yaml [--push]
# build-queries (login node, CPU, minutes) -> generate (GPU array) -> score (GPU array) -> assemble (CPU)
set -euo pipefail
source "$(dirname "$0")/config.sh"
llmr_activate
CFG="$1"; PUSH="${2:-}"
PROTOCOL=$(python - "$CFG" <<'PYCFG'
import sys
from llm_routing.config import load_benchmark_config
print(load_benchmark_config(sys.argv[1]).get("protocol", "legacy"))
PYCFG
)
if [ "$PROTOCOL" = shared-v1 ]; then
    echo "For member collection use submit_member.sh; freeze/merge separately with llmr assemble." >&2
    exit 1
fi
N=$(python -c "from llm_routing.config import load_benchmark_config as l; print(len(l('$CFG')['models']))")
mkdir -p logs
[ -f "$(python -c "from llm_routing.config import load_benchmark_config as l; print(l('$CFG')['path'])")/queries.jsonl" ] || llmr build-queries "$CFG"
COMMON=(--partition="$LLMR_PARTITION" ${LLMR_ACCOUNT:+--account=$LLMR_ACCOUNT} ${LLMR_QOS:+--qos=$LLMR_QOS} ${LLMR_EXCLUDE:+--exclude=$LLMR_EXCLUDE})
GEN=$(sbatch --parsable "${COMMON[@]}" --gres="$(llmr_gres 1)" --array=0-$((N-1)) cluster/tosun/jobs/generate.sbatch "$CFG")
SCO=$(sbatch --parsable "${COMMON[@]}" --gres="$(llmr_gres 1)" --array=0-$((N-1)) --dependency=afterok:$GEN cluster/tosun/jobs/score.sbatch "$CFG")
ASM=$(sbatch --parsable "${COMMON[@]}" --dependency=afterok:$SCO cluster/tosun/jobs/assemble.sbatch "$CFG" $PUSH)
echo "submitted: generate=$GEN score=$SCO assemble=$ASM   (watch with: squeue -u $USER)"
