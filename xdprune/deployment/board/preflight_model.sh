#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <model_name>" >&2
    exit 1
fi

ROOT="${YOLO26_PS_ROOT:-/home/xilinx/yolo26_ps}"
MODEL_NAME="$1"
MODEL_DIR="${ROOT}/1_models/${MODEL_NAME}"
RUNNER="${ROOT}/1_scripts/build/yolo26_ncnn_runner"

if [[ ! "${MODEL_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid model name." >&2
    exit 1
fi

for path in \
    "${MODEL_DIR}/model.ncnn.param" \
    "${MODEL_DIR}/model.ncnn.bin" \
    "${MODEL_DIR}/metadata.yaml" \
    "${ROOT}/1_dataset/val_images.txt" \
    "${ROOT}/1_dataset/benchmark_images.txt"; do
    [[ -s "${path}" ]] || { echo "Missing or empty: ${path}" >&2; exit 1; }
done

[[ -x "${RUNNER}" ]] || { echo "Runner missing or not executable: ${RUNNER}" >&2; exit 1; }

echo "Preflight passed for ${MODEL_NAME}"
echo "Board: $(hostname) / $(uname -m)"
echo "Validation images: $(grep -cve '^[[:space:]]*$' "${ROOT}/1_dataset/val_images.txt")"
echo "Benchmark images: $(grep -cve '^[[:space:]]*$' "${ROOT}/1_dataset/benchmark_images.txt")"
sha256sum \
    "${MODEL_DIR}/model.ncnn.param" \
    "${MODEL_DIR}/model.ncnn.bin" \
    "${MODEL_DIR}/metadata.yaml" \
    "${ROOT}/1_dataset/val_images.txt" \
    "${ROOT}/1_dataset/benchmark_images.txt" \
    "${RUNNER}"
