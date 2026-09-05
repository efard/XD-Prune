#!/usr/bin/env bash
set -Eeuo pipefail

# ---------------------------------------------------------------------------
# /models/
#   0_yolo26n_official/
#   1_baseline_light/
#   2_baseline_full/
#   3_L1_pruned/
#   4_layer_replacement/
# ---------------------------------------------------------------------------

MODEL_ROOT="${MODEL_ROOT:-/home/afm176/yolo_project/models}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/afm176/yolo_project}"

SCRIPT_DIR="${SCRIPT_DIR:-$PROJECT_ROOT/1_scripts/ncnn_accuracy}"

GEN_DATA="${GEN_DATA:-/project/ko/afm176/datasets/1_data/reproduction_exact_v1/GEN_MIO_TCD_exact/dataset_GEN_local.yaml}"

SNOW_DATA="${SNOW_DATA:-/project/ko/afm176/datasets/1_data/reproduction_exact_v1/SNOW_ACDC_exact/dataset_SNOW_local.yaml}"

RESULT_PARENT="${RESULT_PARENT:-$PROJECT_ROOT/5_reproduction/ncnn_accuracy_results}"

IMAGE="${IMAGE:-/opt/software/apptainer-images/pytorch-25.06.sif}"

PACKAGE_PATH="${PACKAGE_PATH:-/project/ko/afm176/yolo-packages}"
NCNN_PYTHON_PATH="${NCNN_PYTHON_PATH:-/project/ko/afm176/python_packages_ncnn}"

WORKERS="${WORKERS:-4}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$RESULT_PARENT/sweep_$STAMP"

mkdir -p "$OUT"

if [[ ! -d "$MODEL_ROOT" ]]; then
  echo "MODEL_ROOT not found: $MODEL_ROOT" >&2
  echo >&2
  echo "Set it when running, for example:" >&2
  echo 'MODEL_ROOT="/home/afm176/yolo_project/1_models" bash ...' >&2
  exit 1
fi

echo "===== NCNN ACCURACY SWEEP ====="
echo "Models: $MODEL_ROOT"
echo "Output: $OUT"
echo "GEN:    $GEN_DATA"
echo "SNOW:   $SNOW_DATA"
echo

module load apptainer

set +e
set +o pipefail

apptainer exec \
  -B /home/afm176:/home/afm176 \
  -B /project/ko:/project/ko \
  --env PYTHONPATH="$NCNN_PYTHON_PATH:$PACKAGE_PATH" \
  "$IMAGE" \
  python "$SCRIPT_DIR/evaluate_all_ncnn.py" \
    --model-root "$MODEL_ROOT" \
    --gen-data "$GEN_DATA" \
    --snow-data "$SNOW_DATA" \
    --output-dir "$OUT" \
    --workers "$WORKERS" \
  2>&1 | tee "$OUT/full_run.log"

APP_RC="${PIPESTATUS[0]}"

set -o pipefail
set -e

SUMMARY="$OUT/ncnn_accuracy_summary.json"

if [[ -f "$SUMMARY" ]] && \
   grep -q '"status": "PASSED_NCNN_ACCURACY_SWEEP"' "$SUMMARY"
then
  ln -sfn "$OUT" "$RESULT_PARENT/latest"

  # Include the full log in a second final ZIP after the evaluator finishes.
  (
    cd "$OUT"
    zip -q \
      "NCNN_accuracy_results_with_log.zip" \
      ncnn_accuracy_results.csv \
      ncnn_accuracy_ranked_by_map50_95.csv \
      ncnn_accuracy_per_class.csv \
      ncnn_accuracy_summary.json \
      README_RESULTS.txt \
      full_run.log
  )

  echo
  echo "===== SUCCESS ====="
  echo "Results:"
  echo "$OUT/ncnn_accuracy_results.csv"
  echo
  echo "Download ZIP:"
  echo "$OUT/NCNN_accuracy_results_with_log.zip"

  exit 0
fi

echo
echo "Sweep did not produce a passing summary." >&2
echo "Apptainer exit code: $APP_RC" >&2
echo "Partial results may still exist at:" >&2
echo "$OUT/ncnn_accuracy_results.csv" >&2
exit 1
