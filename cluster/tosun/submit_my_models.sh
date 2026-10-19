#!/bin/bash
# Usage: bash cluster/tosun/submit_my_models.sh CONFIG MEMBER [NUM_SHARDS]
# Uses CONFIG's collection_team (defaults to cluster/tosun/team.yaml for v1).
# Pilot first:  bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_pilot.yaml <member> 1
set -euo pipefail
source "$(dirname "$0")/config.sh"
llmr_activate
CFG="$1"; MEMBER="$2"; NUM_SHARDS="${3:-1}"
MODELS=$(python - "$CFG" "$MEMBER" <<'PY'
import sys, yaml
from llm_routing.config import load_benchmark_config
cfg = load_benchmark_config(sys.argv[1])
team_path = cfg.get("collection_team", "cluster/tosun/team.yaml")
team = yaml.safe_load(open(team_path))["members"]
mine = [m for m in team if m["member"] == sys.argv[2]]
if not mine:
    sys.exit(f"unknown member {sys.argv[2]!r}; members in {team_path}: {[m['member'] for m in team]}")
print(" ".join(mine[0]["models"]))
PY
)
for MODEL in $MODELS; do
    echo "== $MODEL"
    bash cluster/tosun/submit_member.sh "$CFG" "$MODEL" "$MEMBER" "$NUM_SHARDS"
done
