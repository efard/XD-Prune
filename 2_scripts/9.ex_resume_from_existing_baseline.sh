#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/afm176/yolo_project}"
SCRIPT_DIR="${SCRIPT_DIR:-$PROJECT_ROOT/1_scripts/acdc_pure_global_l1_56p81}"
RESULT_PARENT="${RESULT_PARENT:-$PROJECT_ROOT/5_reproduction/acdc_pure_global_l1_56p81_results}"
IMAGE="${APPTAINER_IMAGE:-/opt/software/apptainer-images/pytorch-25.06.sif}"
PACKAGE_PATH="${PACKAGE_PATH:-/project/ko/afm176/yolo-packages}"

TARGET_PERCENT="${TARGET_PERCENT:-56.81}"
RECOVERY_EPOCHS="${RECOVERY_EPOCHS:-20}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-0}"

# By default resume the newest run that already has the trained ACDC baseline.
if [[ -z "${RUN_DIR:-}" ]]; then
  RUN_DIR="$(find "$RESULT_PARENT" -maxdepth 1 -type d -name 'run_*' \
    -exec test -f '{}/baseline/ACDC_YOLO26n_baseline_best.pt' ';' -print \
    | sort | tail -1)"
fi

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "ERROR: no existing run with baseline checkpoint found." >&2
  exit 1
fi

BASELINE_MODEL="$RUN_DIR/baseline/ACDC_YOLO26n_baseline_best.pt"
if [[ ! -f "$BASELINE_MODEL" ]]; then
  echo "ERROR: baseline checkpoint missing: $BASELINE_MODEL" >&2
  exit 1
fi

if [[ -f "$RUN_DIR/dataset_evidence/selected_dataset_yaml.txt" ]]; then
  DATA_YAML="$(tr -d '\r\n' < "$RUN_DIR/dataset_evidence/selected_dataset_yaml.txt")"
else
  DATA_YAML="${DATA_YAML:-/project/ko/afm176/datasets/1_data/reproduction_exact_v1/SNOW_ACDC_exact/dataset_SNOW_local.yaml}"
fi

if [[ ! -f "$DATA_YAML" ]]; then
  echo "ERROR: dataset YAML missing: $DATA_YAML" >&2
  exit 1
fi

module load apptainer

OUT="$RUN_DIR/pure_global_l1"
rm -rf "$OUT"
mkdir -p "$OUT"

echo "===== RESUME FROM EXISTING BASELINE ====="
echo "Run dir:       $RUN_DIR"
echo "Baseline:      $BASELINE_MODEL"
echo "Dataset:       $DATA_YAML"
echo "Target:        $TARGET_PERCENT%"
echo "Recovery:      $RECOVERY_EPOCHS epochs"
echo

apptainer exec --nv \
  -B /home/afm176:/home/afm176 \
  -B /project/ko:/project/ko \
  --env PYTHONPATH="$PACKAGE_PATH" \
  "$IMAGE" \
  python "$SCRIPT_DIR/03_pure_global_l1_prune_recover.py" \
    --baseline-model "$BASELINE_MODEL" \
    --data "$DATA_YAML" \
    --output-dir "$OUT" \
    --target-percent "$TARGET_PERCENT" \
    --epochs "$RECOVERY_EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --seed "$SEED" \
    --device "$DEVICE"

SUMMARY="$OUT/final_summary.json"
if [[ ! -f "$SUMMARY" ]] || ! grep -q 'PASSED_ACDC_PURE_GLOBAL_L1_56P81' "$SUMMARY"; then
  echo "ERROR: Stage 3 did not finish successfully." >&2
  exit 1
fi

# Package without re-running dataset preparation or baseline training.
BUNDLE="$RUN_DIR/ACDC_pure_GlobalL1_56p81_bundle"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/models" "$BUNDLE/results" "$BUNDLE/scripts" "$BUNDLE/dataset"

cp "$BASELINE_MODEL" "$BUNDLE/models/"
cp "$OUT/ACDC_pure_GlobalL1_56p81_raw.pt" "$BUNDLE/models/"
cp "$OUT/ACDC_pure_GlobalL1_56p81_recovered_best.pt" "$BUNDLE/models/"

for f in raw_validation.csv best_validation.csv summary_results.csv final_summary.json recovery_args.yaml recovery_training_results.csv; do
  [[ -f "$OUT/$f" ]] && cp "$OUT/$f" "$BUNDLE/results/"
done
for f in chosen_global_l1_plan.csv global_l1_selection_log.csv dependency_operations.csv rejected_candidates.csv dependency_safe_generic_scope.csv search_summary.json eligible_roots.txt protected_roots.txt; do
  [[ -f "$OUT/search/$f" ]] && cp "$OUT/search/$f" "$BUNDLE/results/"
done
for f in baseline_validation.csv baseline_summary.json baseline_training_args.yaml baseline_training_results.csv; do
  [[ -f "$RUN_DIR/baseline/$f" ]] && cp "$RUN_DIR/baseline/$f" "$BUNDLE/results/"
done
for f in dataset_report.json train_manifest.txt val_manifest.txt selected_dataset_yaml.txt; do
  [[ -f "$RUN_DIR/dataset_evidence/$f" ]] && cp "$RUN_DIR/dataset_evidence/$f" "$BUNDLE/dataset/"
done

cp "$SCRIPT_DIR/01_prepare_acdc_400_100.py" "$BUNDLE/scripts/"
cp "$SCRIPT_DIR/02_train_acdc_yolo26n.py" "$BUNDLE/scripts/"
cp "$SCRIPT_DIR/03_pure_global_l1_prune_recover.py" "$BUNDLE/scripts/"
cp "$SCRIPT_DIR/run_all.sh" "$BUNDLE/scripts/"
cp "$SCRIPT_DIR/resume_stage3_from_existing_baseline.sh" "$BUNDLE/scripts/"

cat > "$BUNDLE/README.txt" <<EOF
ACDC YOLO26n Pure Global-L1 56.81%
==================================

This run uses raw Global-L1 output-channel magnitude ranking over the 42
prevalidated GENERIC dependency-group representative roots. There is no per-root
channel cap, no minimum-4 rule, no sensitivity weighting, no accuracy-aware
ranking, and no manual layer priority. The 9 CUSTOM C3k2/C2PSA groups are
excluded because their custom pruning implementation is not present. The
dependency-group scope is structural; it is not a pruning-ratio heuristic.

Dataset: $DATA_YAML
Target parameter reduction: $TARGET_PERCENT%
Recovery epochs: $RECOVERY_EPOCHS
Seed: $SEED
EOF

FINAL_ZIP="$RUN_DIR/ACDC_pure_GlobalL1_56p81_results.zip"
rm -f "$FINAL_ZIP"
(
  cd "$RUN_DIR"
  zip -rq "$(basename "$FINAL_ZIP")" "$(basename "$BUNDLE")"
)

ln -sfn "$RUN_DIR" "$RESULT_PARENT/latest"

echo
echo "===== COMPLETE ====="
echo "Raw model:    $OUT/ACDC_pure_GlobalL1_56p81_raw.pt"
echo "Best model:   $OUT/ACDC_pure_GlobalL1_56p81_recovered_best.pt"
echo "Results ZIP:  $FINAL_ZIP"
