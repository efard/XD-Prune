#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <model_name> <benchmark|validate|all> [image_limit]" >&2
    exit 1
fi

MODEL_NAME="$1"
MODE="$2"
LIMIT="${3:-0}"
ROOT="${YOLO26_PS_ROOT:-/home/xilinx/yolo26_ps}"
IMGSZ="${IMGSZ:-640}"
THREADS=2
WARMUP=5
REPEAT=3
CONF_BENCH=0.25
CONF_VALIDATE=0.001
IOU=0.70
MAX_DET=300
RUNNER="${ROOT}/1_scripts/build/yolo26_ncnn_runner"
MODEL_DIR="${ROOT}/1_models/${MODEL_NAME}"
VAL_LIST="${ROOT}/1_dataset/val_images.txt"
BENCH_LIST="${ROOT}/1_dataset/benchmark_images.txt"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="${ROOT}/2_results/${MODEL_NAME}__${STAMP}"

[[ "${MODEL_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid model name" >&2; exit 1; }
[[ "${MODE}" =~ ^(benchmark|validate|all)$ ]] || { echo "Invalid mode" >&2; exit 1; }
[[ "${LIMIT}" =~ ^[0-9]+$ ]] || { echo "image_limit must be a non-negative integer" >&2; exit 1; }

bash "${ROOT}/1_scripts/preflight_model.sh" "${MODEL_NAME}"
mkdir -p "${RESULT_DIR}"

sha256sum \
    "${MODEL_DIR}/model.ncnn.param" \
    "${MODEL_DIR}/model.ncnn.bin" \
    "${MODEL_DIR}/metadata.yaml" \
    "${VAL_LIST}" "${BENCH_LIST}" "${RUNNER}" "$0" \
    > "${RESULT_DIR}/artifact_hashes.txt"
cp "${MODEL_DIR}/metadata.yaml" "${RESULT_DIR}/metadata.yaml"
[[ ! -f "${MODEL_DIR}/deployment_manifest.json" ]] || cp "${MODEL_DIR}/deployment_manifest.json" "${RESULT_DIR}/deployment_manifest.json"

PARAM_BYTES="$(stat -c %s "${MODEL_DIR}/model.ncnn.param")"
BIN_BYTES="$(stat -c %s "${MODEL_DIR}/model.ncnn.bin")"
printf 'model,param_bytes,bin_bytes,total_bytes,total_mib\n%s,%s,%s,%s,%.8f\n' \
    "${MODEL_NAME}" "${PARAM_BYTES}" "${BIN_BYTES}" "$((PARAM_BYTES + BIN_BYTES))" \
    "$(awk -v n="$((PARAM_BYTES + BIN_BYTES))" 'BEGIN {print n/1024/1024}')" \
    > "${RESULT_DIR}/model_size.csv"

{
    echo "status=started"
    echo "date_utc=${STAMP}"
    echo "hostname=$(hostname)"
    echo "kernel=$(uname -a)"
    echo "architecture=$(uname -m)"
    echo "logical_cores=$(nproc)"
    echo "model=${MODEL_NAME}"
    echo "mode=${MODE}"
    echo "image_limit=${LIMIT}"
    echo "imgsz=${IMGSZ}"
    echo "threads=${THREADS}"
    echo "warmup=${WARMUP}"
    echo "repeat=${REPEAT}"
    echo "benchmark_conf=${CONF_BENCH}"
    echo "validation_conf=${CONF_VALIDATE}"
    echo "iou=${IOU}"
    echo "max_det=${MAX_DET}"
} > "${RESULT_DIR}/run_config.txt"

LIMIT_ARGS=()
[[ "${LIMIT}" -eq 0 ]] || LIMIT_ARGS=(--limit "${LIMIT}")

run_validate() {
    "${RUNNER}" --mode validate --model-dir "${MODEL_DIR}" \
        --image-list "${VAL_LIST}" --imgsz "${IMGSZ}" --threads "${THREADS}" \
        --conf "${CONF_VALIDATE}" --iou "${IOU}" --max-det "${MAX_DET}" \
        --predictions "${RESULT_DIR}/predictions.csv" \
        --images-summary "${RESULT_DIR}/images.csv" "${LIMIT_ARGS[@]}"
}

run_benchmark() {
    "${RUNNER}" --mode benchmark --model-dir "${MODEL_DIR}" \
        --image-list "${BENCH_LIST}" --imgsz "${IMGSZ}" --threads "${THREADS}" \
        --warmup "${WARMUP}" --repeat "${REPEAT}" --conf "${CONF_BENCH}" \
        --iou "${IOU}" --max-det "${MAX_DET}" \
        --summary "${RESULT_DIR}/performance.csv" "${LIMIT_ARGS[@]}"
}

case "${MODE}" in
    validate) run_validate ;;
    benchmark) run_benchmark ;;
    all) run_validate; run_benchmark ;;
esac

sed -i 's/^status=started$/status=completed/' "${RESULT_DIR}/run_config.txt"
ARCHIVE="${RESULT_DIR}.tar.gz"
tar -czf "${ARCHIVE}" -C "${ROOT}/2_results" "$(basename "${RESULT_DIR}")"
echo "Completed result: ${RESULT_DIR}"
echo "Archive for transfer: ${ARCHIVE}"
