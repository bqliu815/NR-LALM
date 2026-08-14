#!/bin/bash
#SBATCH --job-name=nrlalm-s52
#SBATCH --exclusive
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-14

set -euo pipefail

EXPERIMENT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_EXE=${PYTHON_EXE:-python3}
CONFIG=${CONFIG:-"$EXPERIMENT_DIR/configs/paper_benchmark.json"}
RESULTS_ROOT=${RESULTS_ROOT:-"$EXPERIMENT_DIR/results/paper_run"}

"$PYTHON_EXE" "$EXPERIMENT_DIR/scripts/run_dataset.py" \
  --config "$CONFIG" \
  --results-root "$RESULTS_ROOT" \
  --dataset-index "${SLURM_ARRAY_TASK_ID:?This script requires a Slurm array index}"
