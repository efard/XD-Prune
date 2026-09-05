#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash 1_scripts/run_model.sh baseline_full validate
#   bash 1_scripts/run_model.sh baseline_full benchmark
#   bash 1_scripts/run_model.sh baseline_full all
#   bash 1_scripts/run_model.sh baseline_full benchmark 1
#
# The optional fourth argument limits the number of images and is intended
# only for a smoke test. Omit it for the real experiment.

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <model_name> <validate|benchmark|all> [image_limit]" >&2
    exit 1
fi

MODEL_NAME="$1"
MODE="$2"
IMAGE_LIMIT="${3:-0}"
IMGSZ="${IMGSZ:-640}"

ROOT="/home/xilinx/yolo26_ps"
RUNNER="${ROOT}/1_scripts/build/yolo26_ncnn_runner"
MODEL_DIR="${ROOT}/1_models/${MODEL_NAME}"
RESULT_DIR="${ROOT}/2_results/${MODEL_NAME}"
VAL_LIST="${ROOT}/1_dataset/val_images.txt"
BENCHMARK_LIST="${ROOT}/1_dataset/benchmark_images.txt"

case "${MODE}" in
    validate|benchmark|all)
        ;;
    *)
        echo "Mode must be validate, benchmark, or all." >&2
        exit 1
        ;;
esac

if [[ ! -x "${RUNNER}" ]]; then
    echo "Runner not found or not executable: ${RUNNER}" >&2
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

if ! [[ "${IMAGE_LIMIT}" =~ ^[0-9]+$ ]]; then
    echo "image_limit must be a non-negative integer." >&2
    exit 1
fi

mkdir -p "${RESULT_DIR}"

# Record the actual deployment size as param + bin. This is separate from
# source .pt size, which should be measured on the PC/server.
PARAM_BYTES="$(stat -c %s "${MODEL_DIR}/model.ncnn.param")"
BIN_BYTES="$(stat -c %s "${MODEL_DIR}/model.ncnn.bin")"
TOTAL_BYTES="$((PARAM_BYTES + BIN_BYTES))"

awk \
    -v model="${MODEL_NAME}" \
    -v param="${PARAM_BYTES}" \
    -v bin="${BIN_BYTES}" \
    -v total="${TOTAL_BYTES}" \
    'BEGIN {
        print "model,param_bytes,bin_bytes,total_bytes,total_mib";
        printf "%s,%d,%d,%d,%.8f\n",
               model, param, bin, total, total / 1024 / 1024;
    }' > "${RESULT_DIR}/model_size.csv"

{
    echo "date=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "hostname=$(hostname)"
    echo "kernel=$(uname -a)"
    echo "architecture=$(uname -m)"
    echo "logical_cores=$(nproc)"
    echo "model=${MODEL_NAME}"
    echo "mode=${MODE}"
    echo "image_limit=${IMAGE_LIMIT}"
    echo "imgsz=${IMGSZ}"
    echo "threads=2"
} > "${RESULT_DIR}/run_config.txt"

LIMIT_ARGS=()
if [[ "${IMAGE_LIMIT}" -gt 0 ]]; then
    LIMIT_ARGS=(--limit "${IMAGE_LIMIT}")
fi

run_validate() {
    if [[ ! -s "${VAL_LIST}" ]]; then
        echo "Validation image list missing or empty: ${VAL_LIST}" >&2
        exit 1
    fi

    "${RUNNER}" \
        --mode validate \
        --model-dir "${MODEL_DIR}" \
        --image-list "${VAL_LIST}" \
        --imgsz "${IMGSZ}" \
        --threads 2 \
        --conf 0.001 \
        --iou 0.70 \
        --max-det 300 \
        --predictions "${RESULT_DIR}/predictions.csv" \
        --images-summary "${RESULT_DIR}/images.csv" \
        "${LIMIT_ARGS[@]}"
}

run_benchmark() {
    if [[ ! -s "${BENCHMARK_LIST}" ]]; then
        echo "Benchmark image list missing or empty: ${BENCHMARK_LIST}" >&2
        exit 1
    fi

    "${RUNNER}" \
        --mode benchmark \
        --model-dir "${MODEL_DIR}" \
        --image-list "${BENCHMARK_LIST}" \
        --imgsz "${IMGSZ}" \
        --threads 2 \
        --warmup 5 \
        --repeat 3 \
        --conf 0.25 \
        --iou 0.70 \
        --max-det 300 \
        --summary "${RESULT_DIR}/performance.csv" \
        "${LIMIT_ARGS[@]}"

    # Power cannot be read reliably from this PYNQ-Z2 software environment. template:
#     cat > "${RESULT_DIR}/power.csv" <<EOF
# model,idle_power_w,average_inference_power_w,peak_inference_power_w,dynamic_power_w,energy_j_per_image
# ${MODEL_NAME},,,,,
# EOF
}

case "${MODE}" in
    validate)
        run_validate
        ;;
    benchmark)
        run_benchmark
        ;;
    all)
        run_validate
        run_benchmark
        ;;
esac

echo "Results saved under: ${RESULT_DIR}"
find "${RESULT_DIR}" -maxdepth 1 -type f -printf "%f  %s bytes\n" | sort
