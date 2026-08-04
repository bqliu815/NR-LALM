#!/bin/bash
#SBATCH --job-name=nrlalm-s53
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --array=0-19

set -euo pipefail

EXPERIMENT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_EXE=${PYTHON_EXE:-python3}
CONFIG=${CONFIG:-"$EXPERIMENT_DIR/configs/paper_v1.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"$EXPERIMENT_DIR/results/paper_run/raw"}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=0

"$PYTHON_EXE" "$EXPERIMENT_DIR/scripts/run_repeat.py" \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --array-index "${SLURM_ARRAY_TASK_ID:?This script requires a Slurm array index}"
