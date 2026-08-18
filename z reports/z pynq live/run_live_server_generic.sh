#!/usr/bin/env bash
set -euo pipefail

# Generic launcher for any compatible NCNN YOLO model stored under:
#   /home/xilinx/yolo26_ps/1_models/<MODEL_NAME>/
#
# Required files in each model directory:
#   model.ncnn.param
#   model.ncnn.bin
#   metadata.yaml
#
# The input size is read from metadata.yaml instead of being hard-coded.
# If imgsz is missing, invalid, or non-square, the script stops rather than
# guessing a size, because a wrong input size would make the test unreliable.
#
# Usage:
#   bash 1_scripts/run_live_server.sh 4_layer_replacement
#   bash 1_scripts/run_live_server.sh 4_layer_replacement_320_FP16
#   bash 1_scripts/run_live_server.sh any_future_model_name

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <model_name>" >&2
    exit 1
fi

ROOT="/home/xilinx/yolo26_ps"
MODEL_NAME="$1"
MODEL_DIR="${ROOT}/1_models/${MODEL_NAME}"
SERVER="${ROOT}/1_scripts/build/yolo26_ncnn_server"

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "Model directory not found: ${MODEL_DIR}" >&2
    exit 1
fi

if [[ ! -x "${SERVER}" ]]; then
    echo "Server binary not found: ${SERVER}" >&2
    echo "Build yolo26_ncnn_server first." >&2
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

METADATA="${MODEL_DIR}/metadata.yaml"

# Read imgsz using Python's standard library only, so PyYAML is not required.
# Supported metadata forms include:
#   imgsz:
#   - 320
#   - 320
#
#   imgsz: [320, 320]
#
#   imgsz: 320
IMGSZ="$(
python3 - "${METADATA}" <<'PY'
import re
import sys

metadata_path = sys.argv[1]

with open(metadata_path, "r") as f:
    lines = f.readlines()

values = None

for index, line in enumerate(lines):
    match = re.match(r"^\s*imgsz\s*:\s*(.*?)\s*$", line)
    if not match:
        continue

    same_line = [int(x) for x in re.findall(r"\d+", match.group(1))]

    if same_line:
        values = same_line
    else:
        block_values = []
        for following in lines[index + 1:]:
            item = re.match(r"^\s*-\s*(\d+)\s*$", following)
            if not item:
                break
            block_values.append(int(item.group(1)))
        values = block_values

    break

if not values:
    raise SystemExit("ERROR: Could not read imgsz from metadata.yaml")

if len(values) == 1:
    height = width = values[0]
else:
    height, width = values[0], values[1]

if height <= 0 or width <= 0:
    raise SystemExit("ERROR: imgsz must be positive")

# yolo26_ncnn_server currently accepts one --imgsz value and therefore uses
# a square model input. Reject rectangular metadata instead of silently using
# the wrong shape.
if height != width:
    raise SystemExit(
        "ERROR: This live server currently requires square imgsz, "
        "but metadata contains {}x{}".format(height, width)
    )

print(height)
PY
)"

echo "Starting live inference server"
echo "  model: ${MODEL_NAME}"
echo "  directory: ${MODEL_DIR}"
echo "  imgsz: ${IMGSZ}"

exec "${SERVER}" \
    --model-dir "${MODEL_DIR}" \
    --imgsz "${IMGSZ}" \
    --threads 2 \
    --port 5000 \
    --conf 0.25 \
    --iou 0.70 \
    --max-det 300
