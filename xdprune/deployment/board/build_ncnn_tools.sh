#!/usr/bin/env bash
set -euo pipefail

ROOT="${YOLO26_PS_ROOT:-/home/xilinx/yolo26_ps}"
SOURCE_DIR="${ROOT}/1_scripts/src"
BUILD_DIR="${ROOT}/1_scripts/build"
CXX="${CXX:-g++}"
mkdir -p "${BUILD_DIR}"

for source in yolo26_ncnn_runner.cpp yolo26_ncnn_server.cpp; do
    [[ -s "${SOURCE_DIR}/${source}" ]] || { echo "Missing source: ${SOURCE_DIR}/${source}" >&2; exit 1; }
done
command -v "${CXX}" >/dev/null || { echo "Compiler not found: ${CXX}" >&2; exit 1; }
command -v pkg-config >/dev/null || { echo "pkg-config is required" >&2; exit 1; }
pkg-config --exists opencv4 || { echo "opencv4 pkg-config entry is missing" >&2; exit 1; }
pkg-config --exists ncnn || {
    echo "ncnn pkg-config entry is missing. Configure a valid NCNN install prefix/build command before continuing." >&2
    exit 1
}

COMMON=(-O3 -std=c++17 -pthread)
read -r -a OPENCV_FLAGS <<< "$(pkg-config --cflags --libs opencv4)"
read -r -a NCNN_FLAGS <<< "$(pkg-config --cflags --libs ncnn)"

"${CXX}" "${COMMON[@]}" "${SOURCE_DIR}/yolo26_ncnn_runner.cpp" \
    -o "${BUILD_DIR}/yolo26_ncnn_runner" "${OPENCV_FLAGS[@]}" "${NCNN_FLAGS[@]}"
"${CXX}" "${COMMON[@]}" "${SOURCE_DIR}/yolo26_ncnn_server.cpp" \
    -o "${BUILD_DIR}/yolo26_ncnn_server" "${OPENCV_FLAGS[@]}" "${NCNN_FLAGS[@]}"

sha256sum "${SOURCE_DIR}"/*.cpp "${BUILD_DIR}/yolo26_ncnn_runner" \
    "${BUILD_DIR}/yolo26_ncnn_server" > "${BUILD_DIR}/build_hashes.txt"
echo "Built NCNN runner and live server under ${BUILD_DIR}"
