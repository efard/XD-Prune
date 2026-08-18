#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/afm176/yolo_project}"
SCRIPT_DIR="${SCRIPT_DIR:-$PROJECT_ROOT/1_scripts/global_l1_stage5c}"
RESULT_PARENT="${RESULT_PARENT:-$PROJECT_ROOT/5_reproduction/global_l1_stage5c_results}"
IMAGE="${IMAGE:-/opt/software/apptainer-images/pytorch-25.06.sif}"
PYTHONPATH_SERVER="${PYTHONPATH_SERVER:-/project/ko/afm176/yolo-packages}"

REPRO="$PROJECT_ROOT/5_reproduction/reproducing_files_for_hoyin"
SEARCH_DIR="${SEARCH_DIR:-$PROJECT_ROOT/5_reproduction/global_l1_stage5b_v2_results/latest}"

GEN_MODEL="${GEN_MODEL:-$REPRO/models/GEN_baseline_best.pt}"
SNOW_MODEL="${SNOW_MODEL:-$REPRO/models/SNOW_baseline_best.pt}"
T4="$REPRO/tables/T4_25pct_group_ranking.csv"
PROTECTED="$REPRO/group_definitions/protected_root_manifest.csv"

GEN_DATA_ROOT="${GEN_DATA_ROOT:-/project/ko/afm176/datasets/1_data/reproduction_exact_v1/GEN_MIO_TCD_exact}"
SNOW_DATA_ROOT="${SNOW_DATA_ROOT:-/project/ko/afm176/datasets/1_data/reproduction_exact_v1/SNOW_ACDC_exact}"

resolve_yaml() {
  local root="$1"

  for candidate in \
    "$root/data.yaml" \
    "$root/dataset.yaml" \
    "$root/data.yml" \
    "$root/dataset.yml"
  do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  local found
  found="$(
    find "$root" -maxdepth 2 -type f \
      \( -name '*.yaml' -o -name '*.yml' \) \
      | sort \
      | head -n 1
  )"

  if [[ -n "$found" ]]; then
    printf '%s\n' "$found"
    return 0
  fi

  return 1
}

GEN_DATA="${GEN_DATA:-$(resolve_yaml "$GEN_DATA_ROOT")}"
SNOW_DATA="${SNOW_DATA:-$(resolve_yaml "$SNOW_DATA_ROOT")}"

for file in \
  "$GEN_MODEL" \
  "$SNOW_MODEL" \
  "$GEN_DATA" \
  "$SNOW_DATA" \
  "$SEARCH_DIR/stage5b_v2_summary.json" \
  "$SEARCH_DIR/GEN/chosen_replay_plan.csv" \
  "$SEARCH_DIR/SNOW/chosen_replay_plan.csv" \
  "$T4" \
  "$PROTECTED"
do
  if [[ ! -f "$file" ]]; then
    echo "Required file not found: $file" >&2
    exit 1
  fi
done

echo "GEN data YAML:  $GEN_DATA"
echo "SNOW data YAML: $SNOW_DATA"
echo "Search input:   $SEARCH_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$RESULT_PARENT/raw_$STAMP"
LOG_DIR="$PROJECT_ROOT/logs/global_l1_stage5c"
LOG="$LOG_DIR/raw_$STAMP.log"

mkdir -p "$OUT" "$LOG_DIR"

module load apptainer

apptainer exec --nv \
  -B /home/afm176:/home/afm176 \
  -B /project/ko:/project/ko \
  --env PYTHONPATH="$PYTHONPATH_SERVER" \
  "$IMAGE" \
  python "$SCRIPT_DIR/06_replay_save_reload_validate_raw.py" \
    --gen-model "$GEN_MODEL" \
    --snow-model "$SNOW_MODEL" \
    --gen-data "$GEN_DATA" \
    --snow-data "$SNOW_DATA" \
    --search-dir "$SEARCH_DIR" \
    --t4 "$T4" \
    --protected-manifest "$PROTECTED" \
    --output-dir "$OUT" \
    --imgsz 640 \
    --batch 16 \
    --workers 4 \
    --seed 42 \
    --score-tolerance 1e-5 \
    --forward-check-interval 25 \
  2>&1 | tee "$LOG"

ln -sfn "$OUT" "$RESULT_PARENT/latest"

echo
echo "===== STAGE 5C SUMMARY ====="
cat "$OUT/stage5c_summary.json"
