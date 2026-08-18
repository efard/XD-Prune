#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/afm176/yolo_project}"
SCRIPT_DIR="${SCRIPT_DIR:-$PROJECT_ROOT/1_scripts/gen50_efficiency_greedy}"
RESULT_PARENT="${RESULT_PARENT:-$PROJECT_ROOT/5_reproduction/layer_replacement_50_efficiency_greedy_gen_results}"

IMAGE="${IMAGE:-/opt/software/apptainer-images/pytorch-25.06.sif}"
PACKAGE_PATH="${PACKAGE_PATH:-/project/ko/afm176/yolo-packages}"

GEN_BASELINE="${GEN_BASELINE:-$PROJECT_ROOT/5_reproduction/reproducing_files_for_hoyin/models/GEN_baseline_best.pt}"
GEN_DATA="${GEN_DATA:-/project/ko/afm176/datasets/1_data/reproduction_exact_v1/GEN_MIO_TCD_exact/dataset_GEN_local.yaml}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$RESULT_PARENT/GEN_efficiency_greedy_$STAMP"
LOG_DIR="$PROJECT_ROOT/logs/gen50_efficiency_greedy"
LOG="$LOG_DIR/GEN_efficiency_greedy_$STAMP.log"

mkdir -p "$OUT" "$LOG_DIR"

module load apptainer

set +e
set +o pipefail

apptainer exec --nv \
  -B /home/afm176:/home/afm176 \
  -B /project/ko:/project/ko \
  --env PYTHONPATH="$SCRIPT_DIR:$PACKAGE_PATH" \
  "$IMAGE" \
  python "$SCRIPT_DIR/run_gen50_efficiency_greedy.py" \
    --project-root "$PROJECT_ROOT" \
    --baseline-model "$GEN_BASELINE" \
    --data "$GEN_DATA" \
    --out-dir "$OUT" \
    --target-percent 50 \
    --epochs 20 \
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
   grep -q '"status": "PASSED_GEN50_EFFICIENCY_GREEDY_FULL"' "$SUMMARY"
then
  ln -sfn "$OUT" "$RESULT_PARENT/latest"

  echo
  echo "===== SUCCESS ====="
  cat "$SUMMARY"

  if [[ "$APP_RC" -ne 0 ]]; then
    echo
    echo "Process exit code $APP_RC occurred after saved outputs passed."
  fi

  echo
  echo "Final model:"
  echo "$OUT/GEN_layer_replacement_efficiency_greedy_recovered_best.pt"
  echo
  echo "Bundle:"
  echo "$OUT/GEN_layer_replacement_efficiency_greedy_full_bundle.zip"

  exit 0
fi

echo "GEN efficiency-greedy full run failed." >&2
echo "Apptainer exit code: $APP_RC" >&2
exit 1
