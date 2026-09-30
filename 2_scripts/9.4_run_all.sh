#!/usr/bin/env bash
set -Eeuo pipefail

# EDITABLE VARIABLES ============================================================================
PROJECT_ROOT="${PROJECT_ROOT:-/home/afm176/yolo_project}"
SCRIPT_DIR="${SCRIPT_DIR:-$PROJECT_ROOT/1_scripts/acdc_pure_global_l1_56p81}"

# Official/pretrained YOLO26n source used to train the ACDC baseline.
SOURCE_MODEL="${SOURCE_MODEL:-$PROJECT_ROOT/yolo26n.pt}"

TARGET_PERCENT="${TARGET_PERCENT:-56.81}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-100}"
RECOVERY_EPOCHS="${RECOVERY_EPOCHS:-20}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-0}"

IMAGE="${APPTAINER_IMAGE:-/opt/software/apptainer-images/pytorch-25.06.sif}"
PACKAGE_PATH="${PACKAGE_PATH:-/project/ko/afm176/yolo-packages}"

RESULT_PARENT="${RESULT_PARENT:-$PROJECT_ROOT/5_reproduction/acdc_pure_global_l1_56p81_results}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$RESULT_PARENT/run_$STAMP"
LOG_DIR="$PROJECT_ROOT/logs/acdc_pure_global_l1_56p81"
LOG="$LOG_DIR/run_$STAMP.log"

mkdir -p "$OUT" "$LOG_DIR"

exec > >(tee -a "$LOG") 2>&1

echo "===== ACDC PURE GLOBAL-L1 ~56.81% ====="
echo "Output:            $OUT"
echo "Source model:      $SOURCE_MODEL"
echo "Target reduction:  $TARGET_PERCENT%"
echo "Baseline epochs:   $BASELINE_EPOCHS"
echo "Recovery epochs:   $RECOVERY_EPOCHS"
echo

if [[ ! -f "$SOURCE_MODEL" ]]; then
  echo "ERROR: SOURCE_MODEL not found: $SOURCE_MODEL" >&2
  echo "Set SOURCE_MODEL=/path/to/yolo26n.pt and rerun." >&2
  exit 1
fi

module load apptainer

# STAGE 1 — DATASET ============================================================================
echo
printf '%s\n' "===== STAGE 1: ACDC 400 TRAIN / 100 VAL ====="

apptainer exec --nv \
  -B /home/afm176:/home/afm176 \
  -B /project/ko:/project/ko \
  --env PYTHONPATH="$PACKAGE_PATH" \
  "$IMAGE" \
  python "$SCRIPT_DIR/01_prepare_acdc_400_100.py" \
    --output-dir "$OUT/dataset_evidence" \
    --seed "$SEED"

DATA_YAML="$(tr -d '\r\n' < "$OUT/dataset_evidence/selected_dataset_yaml.txt")"

if [[ ! -f "$DATA_YAML" ]]; then
  echo "ERROR: selected dataset YAML not found: $DATA_YAML" >&2
  exit 1
fi

echo "Selected dataset: $DATA_YAML"

# STAGE 2 — TRAIN BASELINE ============================================================================
echo
printf '%s\n' "===== STAGE 2: TRAIN ACDC YOLO26n BASELINE ====="

apptainer exec --nv \
  -B /home/afm176:/home/afm176 \
  -B /project/ko:/project/ko \
  --env PYTHONPATH="$PACKAGE_PATH" \
  "$IMAGE" \
  python "$SCRIPT_DIR/02_train_acdc_yolo26n.py" \
    --data "$DATA_YAML" \
    --source-model "$SOURCE_MODEL" \
    --output-dir "$OUT/baseline" \
    --epochs "$BASELINE_EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --seed "$SEED"

BASELINE_MODEL="$OUT/baseline/ACDC_YOLO26n_baseline_best.pt"

if [[ ! -f "$BASELINE_MODEL" ]]; then
  echo "ERROR: trained baseline best.pt was not created." >&2
  exit 1
fi

# STAGE 3 — PURE GLOBAL-L1 + RAW VAL + 20E RECOVERY + BEST VAL ============================================================================
echo
printf '%s\n' "===== STAGE 3: PURE GLOBAL-L1 PRUNE / RECOVER / VALIDATE ====="

apptainer exec --nv \
  -B /home/afm176:/home/afm176 \
  -B /project/ko:/project/ko \
  --env PYTHONPATH="$PACKAGE_PATH" \
  "$IMAGE" \
  python "$SCRIPT_DIR/03_pure_global_l1_prune_recover.py" \
    --baseline-model "$BASELINE_MODEL" \
    --data "$DATA_YAML" \
    --output-dir "$OUT/pure_global_l1" \
    --target-percent "$TARGET_PERCENT" \
    --epochs "$RECOVERY_EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --seed "$SEED" \
    --device "$DEVICE"

SUMMARY="$OUT/pure_global_l1/final_summary.json"
if [[ ! -f "$SUMMARY" ]]; then
  echo "ERROR: final_summary.json was not created." >&2
  exit 1
fi

if ! grep -q 'PASSED_ACDC_PURE_GLOBAL_L1_56P81' "$SUMMARY"; then
  echo "ERROR: experiment summary did not report PASS." >&2
  cat "$SUMMARY" >&2
  exit 1
fi

# STAGE 4 — FINAL DOWNLOAD ZIP ============================================================================
echo
printf '%s\n' "===== STAGE 4: PACKAGE RESULTS ====="

BUNDLE="$OUT/ACDC_pure_GlobalL1_56p81_bundle"
rm -rf "$BUNDLE"
mkdir -p \
  "$BUNDLE/models" \
  "$BUNDLE/results" \
  "$BUNDLE/scripts" \
  "$BUNDLE/dataset"

cp "$OUT/baseline/ACDC_YOLO26n_baseline_best.pt" \
  "$BUNDLE/models/"
cp "$OUT/pure_global_l1/ACDC_pure_GlobalL1_56p81_raw.pt" \
  "$BUNDLE/models/"
cp "$OUT/pure_global_l1/ACDC_pure_GlobalL1_56p81_recovered_best.pt" \
  "$BUNDLE/models/"

# Requested results.
cp "$OUT/pure_global_l1/raw_validation.csv" "$BUNDLE/results/"
cp "$OUT/pure_global_l1/best_validation.csv" "$BUNDLE/results/"
cp "$OUT/pure_global_l1/summary_results.csv" "$BUNDLE/results/"
cp "$OUT/pure_global_l1/final_summary.json" "$BUNDLE/results/"
cp "$OUT/pure_global_l1/search/chosen_global_l1_plan.csv" "$BUNDLE/results/"
cp "$OUT/pure_global_l1/search/global_l1_selection_log.csv" "$BUNDLE/results/"
cp "$OUT/pure_global_l1/search/search_summary.json" "$BUNDLE/results/"
cp "$OUT/pure_global_l1/search/dependency_operations.csv" "$BUNDLE/results/" 2>/dev/null || true
cp "$OUT/pure_global_l1/search/rejected_candidates.csv" "$BUNDLE/results/" 2>/dev/null || true
cp "$OUT/pure_global_l1/search/structurally_excluded_roots.csv" "$BUNDLE/results/" 2>/dev/null || true

# Training arguments/results.
cp "$OUT/baseline/baseline_training_args.yaml" "$BUNDLE/results/" 2>/dev/null || true
cp "$OUT/baseline/baseline_training_results.csv" "$BUNDLE/results/" 2>/dev/null || true
cp "$OUT/pure_global_l1/recovery_args.yaml" "$BUNDLE/results/" 2>/dev/null || true
cp "$OUT/pure_global_l1/recovery_training_results.csv" "$BUNDLE/results/" 2>/dev/null || true

# Dataset evidence.
cp "$OUT/dataset_evidence/dataset_report.json" "$BUNDLE/dataset/"
cp "$OUT/dataset_evidence/train_manifest.txt" "$BUNDLE/dataset/"
cp "$OUT/dataset_evidence/val_manifest.txt" "$BUNDLE/dataset/"
printf '%s\n' "$DATA_YAML" > "$BUNDLE/dataset/dataset_yaml_used.txt"

# Used scripts.
cp "$SCRIPT_DIR/01_prepare_acdc_400_100.py" "$BUNDLE/scripts/"
cp "$SCRIPT_DIR/02_train_acdc_yolo26n.py" "$BUNDLE/scripts/"
cp "$SCRIPT_DIR/03_pure_global_l1_prune_recover.py" "$BUNDLE/scripts/"
cp "$SCRIPT_DIR/run_all.sh" "$BUNDLE/scripts/"
cp "$SCRIPT_DIR/resume_stage3_from_existing_baseline.sh" "$BUNDLE/scripts/"

cat > "$BUNDLE/README.txt" <<EOF
ACDC YOLO26n Pure Global-L1 ~56.81%
==================================

Dataset:
- ACDC snow
- 400 train images
- 100 validation images
- dataset YAML: $DATA_YAML

Baseline training:
- YOLO26n
- $BASELINE_EPOCHS epochs
- imgsz $IMGSZ
- batch $BATCH
- AdamW, lr0=0.001, lrf=0.01
- momentum=0.9, weight_decay=0.0005
- warmup=1 epoch
- seed=$SEED, deterministic=True, AMP=True

Pure Global-L1:
- global L1 ranking of eligible Conv2d output channels
- smallest current L1 channel selected globally
- dependency-safe structural pruning with Torch-Pruning
- DepGraph rebuilt after every pruning step
- target whole-model parameter reduction: $TARGET_PERCENT%
- NO per-root/channel pruning cap
- NO minimum-4-channel heuristic
- NO T4 42-root restriction
- NO sensitivity/accuracy-aware selection
- NO layer weighting
- only structurally necessary Detect-output and nonzero-channel constraints
- duplicated YOLO26 Detect input dependencies are synchronized across branches
  without changing L1 candidate scores or global ranking
- depthwise Conv2d uses Torch-Pruning's dedicated depthwise pruning handler
- one2one first-DWConv is dependency-only because x.detach() severs upstream tracing

Recovery:
- $RECOVERY_EPOCHS epochs
- AdamW, lr0=0.001, lrf=0.01
- momentum=0.9, weight_decay=0.0005
- seed=$SEED, deterministic=True, AMP=True

Validation CSV metrics:
- parameter reduction %
- mAP50-95
- mAP50 / mAP75
- Precision / Recall
- average latency ms/image = preprocess + inference + postprocess
- FPS = 1000 / average latency ms
EOF

FINAL_ZIP="$OUT/ACDC_pure_GlobalL1_56p81_results.zip"
(
  cd "$OUT"
  zip -rq "$(basename "$FINAL_ZIP")" "$(basename "$BUNDLE")"
)

ln -sfn "$OUT" "$RESULT_PARENT/latest"

echo
echo "===== COMPLETE ====="
echo "Result folder: $OUT"
echo "Raw model:     $OUT/pure_global_l1/ACDC_pure_GlobalL1_56p81_raw.pt"
echo "Best model:    $OUT/pure_global_l1/ACDC_pure_GlobalL1_56p81_recovered_best.pt"
echo "Raw CSV:       $OUT/pure_global_l1/raw_validation.csv"
echo "Best CSV:      $OUT/pure_global_l1/best_validation.csv"
echo "Download ZIP:  $FINAL_ZIP"
