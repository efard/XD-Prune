#!/usr/bin/env bash
set -Eeuo pipefail

# GEN: fixed 5% subset (550 images, seed 42)
# SNOW: full 100-image validation set
# NCNN: sequential evaluation, 4 threads/model

PROJECT_ROOT="/home/afm176/yolo_project"

MODEL_ROOT="/home/afm176/yolo_project/models"

GEN_DATA="/project/ko/afm176/datasets/1_data/reproduction_exact_v1/GEN_MIO_TCD_exact_val5pct_seed42/dataset_GEN_val5pct_seed42.yaml"

SNOW_DATA="/project/ko/afm176/datasets/1_data/reproduction_exact_v1/SNOW_ACDC_exact/dataset_SNOW_local.yaml"

NCNN_PYTHON_PATH="/project/ko/afm176/python_packages_ncnn_minimal"

V2_RUNNER="$PROJECT_ROOT/1_scripts/ncnn_accuracy_v2/run_ncnn_accuracy.sh"

LOG="$PROJECT_ROOT/logs/ncnn_5pct_last_working.log"
PIDFILE="$PROJECT_ROOT/logs/ncnn_5pct_last_working.pid"

mkdir -p "$PROJECT_ROOT/logs"

if [[ ! -x "$V2_RUNNER" ]]; then
  echo "ERROR: V2 runner not found or not executable:"
  echo "$V2_RUNNER"
  exit 1
fi

if [[ ! -f "$GEN_DATA" ]]; then
  echo "ERROR: 5% GEN subset YAML not found:"
  echo "$GEN_DATA"
  echo
  echo "The previous V4 run should already have created it."
  exit 1
fi

if [[ ! -f "$SNOW_DATA" ]]; then
  echo "ERROR: SNOW YAML not found:"
  echo "$SNOW_DATA"
  exit 1
fi

# APPTAINERENV_* guarantees these variables are visible inside Apptainer
export APPTAINERENV_NCNN_NUM_THREADS=4
export APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_OMP_DYNAMIC=FALSE
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_NUMEXPR_NUM_THREADS=1

# Pass only path/configuration
export MODEL_ROOT
export GEN_DATA
export SNOW_DATA
export NCNN_PYTHON_PATH

cd "$PROJECT_ROOT"

nohup bash "$V2_RUNNER" \
  > "$LOG" 2>&1 &

PID=$!
echo "$PID" | tee "$PIDFILE"
disown

echo
echo "===== STARTED ====="
echo "PID: $PID"
echo "GEN: fixed 550-image subset"
echo "SNOW: full 100-image set"
echo "NCNN threads/model: 4"
echo "Mode: sequential V2"
echo
echo "Monitor:"
echo "tail -f $LOG"
