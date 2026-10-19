#!/bin/bash
# Usage: bash cluster/tosun/submit_my_models.sh CONFIG MEMBER [NUM_SHARDS]
# Runs submit_member.sh once for every model assigned to MEMBER in cluster/tosun/team.yaml.
# Pilot first:  bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_pilot.yaml <member> 1
set -euo pipefail
source "$(dirname "$0")/config.sh"
llmr_activate
CFG="$1"; MEMBER="$2"; NUM_SHARDS="${3:-1}"
MODELS=$(python - "$MEMBER" <<'PY'
import sys, yaml
team = yaml.safe_load(open("cluster/tosun/team.yaml"))["members"]
mine = [m for m in team if m["member"] == sys.argv[1]]
if not mine:
    sys.exit(f"unknown member {sys.argv[1]!r}; members in cluster/tosun/team.yaml: {[m['member'] for m in team]}")
print(" ".join(mine[0]["models"]))
PY
)
for MODEL in $MODELS; do
    echo "== $MODEL"
    bash cluster/tosun/submit_member.sh "$CFG" "$MODEL" "$MEMBER" "$NUM_SHARDS"
done
