#!/bin/bash
# Full 100-epoch YOLO26n 640x640 INT8 QAT run.
# Run this only after the smoke test succeeds and while on an allocated GPU node.

set -euo pipefail

PROJECT_ROOT="/home/afm176/yolo_project"
BASELINE_MODEL="${PROJECT_ROOT}/runs/baseline/baseline_yolo26n_mio_11000_e100_img640_b16_s42_20260701_001856/weights/best.pt"
DATA_YAML="/project/ko/afm176/datasets/1_data/mio_yolo_11000_seed42/mio_tcd.yaml"
APPTAINER_IMG="/opt/software/apptainer-images/pytorch-25.06.sif"
YOLO_PACKAGES="/project/ko/afm176/yolo-packages"
MODELOPT_PACKAGES="/project/ko/afm176/modelopt-packages"
TMP_DIR="/project/ko/afm176/tmp"

RUN_NAME="qat_int8_yolo26n_mio_11000_e100_img640_b16_s42_$(date +%Y%m%d_%H%M%S)"

module load apptainer
cd "${PROJECT_ROOT}"

mkdir -p logs/qat runs/qat "${TMP_DIR}"

apptainer exec --nv \
  -B /project/ko:/project/ko \
  -B /home/afm176:/home/afm176 \
  --env PYTHONPATH="${MODELOPT_PACKAGES}:${YOLO_PACKAGES}" \
  --env TMPDIR="${TMP_DIR}" \
  "${APPTAINER_IMG}" \
  python 1_scripts/4.1_train_yolo26n_qat_int8_640.py \
    --model "${BASELINE_MODEL}" \
    --data "${DATA_YAML}" \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --device 0 \
    --workers 8 \
    --calib-batches 32 \
    --optimizer AdamW \
    --lr0 0.0001 \
    --lrf 0.01 \
    --seed 42 \
    --project runs/qat \
    --name "${RUN_NAME}" \
    --cache false \
  2>&1 | tee "logs/qat/${RUN_NAME}.log"
