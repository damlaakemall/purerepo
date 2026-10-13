#!/bin/bash
# Freeze the benchmark (CPU) and optionally push it to the HF Hub.
#   sbatch cluster/tosun/jobs/assemble.sbatch configs/benchmarks/ours_v1.yaml [--push]
#SBATCH --job-name=llmr-assemble
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source cluster/tosun/config.sh
llmr_activate
llmr assemble "$1"
llmr info "$1"
if [ "${2:-}" = "--push" ]; then llmr push-data "$1"; fi
