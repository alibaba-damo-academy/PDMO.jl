#!/usr/bin/env bash
set -euo pipefail

# Configuration
ROOT="/home/kaizhao-sun/Gitlab/PDMO.jl"
BENCH_NAME="collection/instances"
BENCH_DIR="/home/kaizhao-sun/$BENCH_NAME"
LOG_DIR="$ROOT/applications/GenericLP/logs"

mkdir -p "$LOG_DIR"

# Use 16 threads for Julia
JULIA_T="-t 16"

# Activate project and run a Julia expr
run_julia() {
  local expr="$1"
  julia ${JULIA_T} --color=yes -e "cd(\"$ROOT/applications/GenericLP\"); include(\"LPPipeline.jl\"); $expr"
}

shopt -s nullglob
cases=("$BENCH_DIR"/*.mps.gz)
if [[ ${#cases[@]} -eq 0 ]]; then
  echo "No .mps files found in $BENCH_DIR"
  exit 0
fi

echo "Found ${#cases[@]} cases in $BENCH_DIR"

for mps in "${cases[@]}"; do
  case_name="$(basename "$mps" .mps.gz)"
  echo "Processing: $case_name"

  # 1) Baseline
  run_julia "runLPPipeline(\"$BENCH_NAME/$case_name\")" \
    > "$LOG_DIR/${case_name}.log" 2>&1 || true

  # 2) CoCluster + MILP bipartization
  run_julia "runLPPipelineCoCluster(\"$BENCH_NAME/$case_name\"; bipartizationAlgorithm = MILP_BIPARTIZATION)" \
    > "$LOG_DIR/${case_name}_MILP.log" 2>&1 || true

  # 3) CoCluster + BFS bipartization
  run_julia "runLPPipelineCoCluster(\"$BENCH_NAME/$case_name\"; bipartizationAlgorithm = BFS_BIPARTIZATION)" \
    > "$LOG_DIR/${case_name}_BFS.log" 2>&1 || true
done

echo "All logs written to: $LOG_DIR"


