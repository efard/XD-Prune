#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/afm176/yolo_project}"
SCRIPT_DIR="${SCRIPT_DIR:-$PROJECT_ROOT/1_scripts/global_l1_stage5b_v2}"
RESULT_PARENT="${RESULT_PARENT:-$PROJECT_ROOT/5_reproduction/global_l1_stage5b_v2_results}"
IMAGE="${IMAGE:-/opt/software/apptainer-images/pytorch-25.06.sif}"
PYTHONPATH_SERVER="${PYTHONPATH_SERVER:-/project/ko/afm176/yolo-packages}"

REPRO="$PROJECT_ROOT/5_reproduction/reproducing_files_for_hoyin"

GEN_MODEL="${GEN_MODEL:-$REPRO/models/GEN_baseline_best.pt}"
SNOW_MODEL="${SNOW_MODEL:-$REPRO/models/SNOW_baseline_best.pt}"
T4="$REPRO/tables/T4_25pct_group_ranking.csv"
PROTECTED="$REPRO/group_definitions/protected_root_manifest.csv"

for file in "$GEN_MODEL" "$SNOW_MODEL" "$T4" "$PROTECTED"; do
  if [[ ! -f "$file" ]]; then
    echo "Required file not found: $file" >&2
    exit 1
  fi
done

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$RESULT_PARENT/search_v2_$STAMP"
LOG_DIR="$PROJECT_ROOT/logs/global_l1_stage5b_v2"
LOG="$LOG_DIR/search_v2_$STAMP.log"

mkdir -p "$OUT" "$LOG_DIR"

module load apptainer

apptainer exec --nv \
  -B /home/afm176:/home/afm176 \
  -B /project/ko:/project/ko \
  --env PYTHONPATH="$PYTHONPATH_SERVER" \
  "$IMAGE" \
  python "$SCRIPT_DIR/05_incremental_global_l1_target_search.py" \
    --gen-model "$GEN_MODEL" \
    --snow-model "$SNOW_MODEL" \
    --t4 "$T4" \
    --protected-manifest "$PROTECTED" \
    --output-dir "$OUT" \
    --imgsz 640 \
    --seed 42 \
    --maximum-root-pruning-fraction 0.50 \
    --absolute-minimum-channels 4 \
    --maximum-steps 2000 \
  2>&1 | tee "$LOG"

ln -sfn "$OUT" "$RESULT_PARENT/latest"

echo
echo "===== STAGE 5B V2 SUMMARY ====="
cat "$OUT/stage5b_v2_summary.json"
