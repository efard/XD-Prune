#!/usr/bin/env bash
set -euo pipefail

# Run one of the two models selected for the live-video comparison.
# The input size is explicit for each model so the server never guesses it.
#
# Usage:
#   bash 1_scripts/run_live_server.sh 2_baseline_full
#   bash 1_scripts/run_live_server.sh 4_layer_replacement_320_FP16

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <2_baseline_full|4_layer_replacement_320_FP16>" >&2
    exit 1
fi

ROOT="/home/xilinx/yolo26_ps"
MODEL_NAME="$1"
SERVER="${ROOT}/1_scripts/build/yolo26_ncnn_server"
MODEL_DIR="${ROOT}/1_models/${MODEL_NAME}"

case "${MODEL_NAME}" in
    2_baseline_full)
        IMGSZ=640
        ;;
    4_layer_replacement_320_FP16)
        IMGSZ=320
        ;;
    *)
        echo "Unsupported live-test model: ${MODEL_NAME}" >&2
        exit 1
        ;;
esac

if [[ ! -x "${SERVER}" ]]; then
    echo "Server binary not found: ${SERVER}" >&2
    echo "Build it first with the commands in LIVE_VIDEO_README.txt" >&2
    exit 1
fi

for REQUIRED_FILE in \
    "${MODEL_DIR}/model.ncnn.param" \
    "${MODEL_DIR}/model.ncnn.bin" \
    "${MODEL_DIR}/metadata.yaml"; do
    if [[ ! -s "${REQUIRED_FILE}" ]]; then
        echo "Missing or empty model file: ${REQUIRED_FILE}" >&2
        exit 1
    fi
done

exec "${SERVER}" \
    --model-dir "${MODEL_DIR}" \
    --imgsz "${IMGSZ}" \
    --threads 2 \
    --port 5000 \
    --conf 0.25 \
    --iou 0.70 \
    --max-det 300
