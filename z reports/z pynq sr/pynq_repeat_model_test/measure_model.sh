#!/usr/bin/env bash
set -euo pipefail

# Reusable wrapper for testing one NCNN model on the PYNQ-Z2 PS.
#
# Usage:
#   bash 1_scripts/measure_model.sh <model_name> benchmark
#   bash 1_scripts/measure_model.sh <model_name> validate
#   bash 1_scripts/measure_model.sh <model_name> all
#
# Optional third argument:
#   Number of images to test. Use 1 for a quick smoke test.
#   Omit it for the full experiment.

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <model_name> <benchmark|validate|all> [image_limit]" >&2
    exit 1
fi

MODEL_NAME="$1"
MODE="$2"
IMAGE_LIMIT="${3:-0}"

ROOT="/home/xilinx/yolo26_ps"
MODEL_DIR="${ROOT}/1_models/${MODEL_NAME}"
RESULT_DIR="${ROOT}/2_results/${MODEL_NAME}"
RUN_SCRIPT="${ROOT}/1_scripts/run_model.sh"
RUNNER="${ROOT}/1_scripts/build/yolo26_ncnn_runner"

# Restrict model names to portable directory characters. This prevents spaces
# or shell metacharacters from creating ambiguous result paths.
if [[ ! "${MODEL_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Model name may contain only letters, numbers, dot, underscore, and hyphen." >&2
    exit 1
fi

case "${MODE}" in
    benchmark|validate|all)
        ;;
    *)
        echo "Mode must be benchmark, validate, or all." >&2
        exit 1
        ;;
esac

if ! [[ "${IMAGE_LIMIT}" =~ ^[0-9]+$ ]]; then
    echo "image_limit must be a non-negative integer." >&2
    exit 1
fi

# Stop immediately when a deployment file is missing. Guessing a filename or
# silently using another model could invalidate the experiment.
for REQUIRED_FILE in \
    "${MODEL_DIR}/model.ncnn.param" \
    "${MODEL_DIR}/model.ncnn.bin" \
    "${MODEL_DIR}/metadata.yaml"; do
    if [[ ! -s "${REQUIRED_FILE}" ]]; then
        echo "Missing or empty file: ${REQUIRED_FILE}" >&2
        exit 1
    fi
done

if [[ ! -x "${RUNNER}" ]]; then
    echo "Compiled runner not found: ${RUNNER}" >&2
    exit 1
fi

if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "Benchmark script not found: ${RUN_SCRIPT}" >&2
    exit 1
fi

mkdir -p "${RESULT_DIR}"

# Save hashes so that every result can be traced to the exact model files.
sha256sum \
    "${MODEL_DIR}/model.ncnn.param" \
    "${MODEL_DIR}/model.ncnn.bin" \
    "${MODEL_DIR}/metadata.yaml" \
    > "${RESULT_DIR}/model_hashes.txt"

# Keep the metadata beside the results so the class mapping is not lost when
# the result folder is transferred back to a PC or server.
cp "${MODEL_DIR}/metadata.yaml" "${RESULT_DIR}/metadata.yaml"

echo "============================================================"
echo "Model: ${MODEL_NAME}"
echo "Mode: ${MODE}"
echo "Image limit: ${IMAGE_LIMIT}"
echo "============================================================"

# run_model.sh contains the fixed experimental settings. Passing every model
# through the same script keeps image size, threads, warm-up, and thresholds
# consistent across repeated experiments.
bash "${RUN_SCRIPT}" "${MODEL_NAME}" "${MODE}" "${IMAGE_LIMIT}"

# Package this model's result folder so MobaXterm can download one file without
# mixing CSV files that share the same filename across different models.
ARCHIVE="${ROOT}/2_results/${MODEL_NAME}_results.tar.gz"
tar -czf "${ARCHIVE}" -C "${ROOT}/2_results" "${MODEL_NAME}"

echo
echo "Result folder:"
echo "  ${RESULT_DIR}"
echo
echo "Downloadable archive:"
echo "  ${ARCHIVE}"

# Print the most useful benchmark values when performance.csv exists.
# Python's standard csv module is used because it is available on the board
# and avoids requiring pandas in the PYNQ terminal environment.
if [[ -s "${RESULT_DIR}/performance.csv" ]]; then
    python3 - "${RESULT_DIR}/performance.csv" <<'PY'
import csv
import sys

path = sys.argv[1]

with open(path, newline="") as file:
    row = next(csv.DictReader(file))

print()
print("Key performance results")
print("-----------------------")
print("Model:                    {}".format(row["model"]))
print("Median compute latency:   {:.3f} ms/image".format(
    float(row["median_compute_ms"])
))
print("P95 compute latency:      {:.3f} ms/image".format(
    float(row["p95_compute_ms"])
))
print("Compute FPS:              {:.6f}".format(
    float(row["compute_fps"])
))
print("Peak RSS memory:          {:.3f} MiB".format(
    float(row["peak_rss_mib"])
))
print("Normalized board CPU:     {:.3f}%".format(
    float(row["normalized_board_cpu_percent"])
))
PY
fi

echo
echo "Generated files:"
find "${RESULT_DIR}" \
    -maxdepth 1 \
    -type f \
    -printf "  %f  %s bytes\n" \
    | sort
