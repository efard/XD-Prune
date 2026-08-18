#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/afm176/yolo_project}"
SCRIPT_DIR="${SCRIPT_DIR:-$PROJECT_ROOT/1_scripts/gen56_global_l1_exact_pipeline}"
RESULT_PARENT="${RESULT_PARENT:-$PROJECT_ROOT/5_reproduction/global_l1_56_exact_gen_results}"

IMAGE="${IMAGE:-/opt/software/apptainer-images/pytorch-25.06.sif}"
PACKAGE_PATH="${PACKAGE_PATH:-/project/ko/afm176/yolo-packages}"

GEN_MODEL="${GEN_MODEL:-$PROJECT_ROOT/5_reproduction/reproducing_files_for_hoyin/models/GEN_baseline_best.pt}"
GEN_DATA="${GEN_DATA:-/project/ko/afm176/datasets/1_data/reproduction_exact_v1/GEN_MIO_TCD_exact/dataset_GEN_local.yaml}"

T4="${T4:-$PROJECT_ROOT/5_reproduction/reproducing_files_for_hoyin/tables/T4_25pct_group_ranking.csv}"
PROTECTED="${PROTECTED:-$PROJECT_ROOT/5_reproduction/reproducing_files_for_hoyin/group_definitions/protected_root_manifest.csv}"

TARGET_PERCENT="${TARGET_PERCENT:-56.4956}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$RESULT_PARENT/GEN_global_L1_56_exact_$STAMP"

LOG_DIR="$PROJECT_ROOT/logs/gen56_global_l1_exact_pipeline"
LOG="$LOG_DIR/GEN_global_L1_56_exact_$STAMP.log"

mkdir -p "$OUT" "$LOG_DIR"

echo "===== GEN ~56% GLOBAL L1 EXACT PIPELINE ====="
echo "Output:   $OUT"
echo "Target:   $TARGET_PERCENT%"
echo

module load apptainer

set +e
set +o pipefail

apptainer exec --nv \
  -B /home/afm176:/home/afm176 \
  -B /project/ko:/project/ko \
  --env PYTHONPATH="$SCRIPT_DIR:$PACKAGE_PATH" \
  "$IMAGE" \
  python "$SCRIPT_DIR/run_exact_gen56_pipeline.py" \
    --project-root "$PROJECT_ROOT" \
    --gen-model "$GEN_MODEL" \
    --gen-data "$GEN_DATA" \
    --t4 "$T4" \
    --protected-manifest "$PROTECTED" \
    --output-dir "$OUT" \
    --target-percent "$TARGET_PERCENT" \
    --imgsz 640 \
    --batch 16 \
    --workers 4 \
    --seed 42 \
    --device 0 \
  2>&1 | tee "$LOG"

APP_RC="${PIPESTATUS[0]}"

set -o pipefail
set -e

SUMMARY="$OUT/summary.json"

if [[ -f "$SUMMARY" ]] && \
   grep -q '"status": "PASSED_GEN56_GLOBAL_L1_EXACT_PIPELINE"' "$SUMMARY"
then
  ln -sfn "$OUT" "$RESULT_PARENT/latest"

  echo
  echo "===== SUCCESS ====="
  echo "Recovered best:"
  echo "$OUT/GEN_global_L1_56pct_recovered_best.pt"
  echo
  echo "Download bundle:"
  echo "$OUT/GEN_global_L1_56pct_exact_full_bundle.zip"

  if [[ "$APP_RC" -ne 0 ]]; then
    echo
    echo "Apptainer exit code $APP_RC occurred after a passing summary."
    echo "Accepted as post-completion cleanup."
  fi

  exit 0
fi

echo
echo "GEN ~56% exact Global L1 pipeline did not pass." >&2
echo "Apptainer exit code: $APP_RC" >&2
echo "Log: $LOG" >&2
exit 1
