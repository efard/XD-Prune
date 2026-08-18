#!/usr/bin/env bash
set -Eeuo pipefail

# Package useful Stage 1-4 files for the Stage 5 handoff.
#
# Default behavior:
# - Includes compact Stage 1-3 results.
# - Includes the selected raw L9 and C007 checkpoints.
# - Includes the complete successful Stage 4 recovery folder, including
#   recovered best.pt and last.pt checkpoints.
# - Includes experiment scripts, settings, manifests, checksums, and summaries.
# - Excludes raw datasets and caches.
#
# Optional:
#   INCLUDE_ALL_STAGE2_MODELS=1  Include every Stage 2 .pt file.
#   INCLUDE_ALL_STAGE3_MODELS=1  Include every Stage 3 .pt file.
#   INCLUDE_REPRODUCTION_ZIP=0   Exclude reproducing_files_for_hoyin.zip.
#
# Usage:
#   bash package_stage1_to_stage4_for_stage5.sh
#
# Optional custom output directory:
#   bash package_stage1_to_stage4_for_stage5.sh /home/afm176/yolo_project/6_exports

PROJECT_ROOT="${PROJECT_ROOT:-/home/afm176/yolo_project}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/5_reproduction}"
SCRIPTS_ROOT="${SCRIPTS_ROOT:-$PROJECT_ROOT/1_scripts}"
OUTPUT_PARENT="${1:-$PROJECT_ROOT/6_exports}"

INCLUDE_ALL_STAGE2_MODELS="${INCLUDE_ALL_STAGE2_MODELS:-0}"
INCLUDE_ALL_STAGE3_MODELS="${INCLUDE_ALL_STAGE3_MODELS:-0}"
INCLUDE_REPRODUCTION_ZIP="${INCLUDE_REPRODUCTION_ZIP:-1}"

STAGE2_PREFERRED="$RESULTS_ROOT/stage2_results/layer_replacement_sweep_20260729_151557"
STAGE3_PREFERRED="$RESULTS_ROOT/stage3_results/near_target_20260729_172754"
STAGE4_PREFERRED="$RESULTS_ROOT/stage4_results/recovery_C007_L9_20260729_192450"

timestamp="$(date +%Y%m%d_%H%M%S)"
package_name="YOLO26n_stage1_to_stage4_handoff_${timestamp}"
work_dir="$OUTPUT_PARENT/$package_name"
zip_path="$OUTPUT_PARENT/${package_name}.zip"

mkdir -p "$OUTPUT_PARENT"
rm -rf "$work_dir"
mkdir -p \
  "$work_dir/00_package_information" \
  "$work_dir/01_stage1_baselines_and_environment" \
  "$work_dir/02_stage2_layer_sweep" \
  "$work_dir/03_stage3_candidate_search" \
  "$work_dir/04_stage4_recovery" \
  "$work_dir/05_experiment_scripts" \
  "$work_dir/06_selected_and_referenced_models"

log() {
  printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

latest_dir() {
  local parent="$1"
  if [[ -d "$parent" ]]; then
    find "$parent" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  fi
}

resolve_run_dir() {
  local preferred="$1"
  local parent="$2"
  if [[ -d "$preferred" ]]; then
    printf '%s\n' "$preferred"
  else
    latest_dir "$parent"
  fi
}

copy_file_preserve_name() {
  local src="$1"
  local dest_dir="$2"
  [[ -f "$src" ]] || return 0

  mkdir -p "$dest_dir"

  local base
  base="$(basename "$src")"
  local dest="$dest_dir/$base"

  if [[ -e "$dest" ]]; then
    local stem ext digest
    stem="${base%.*}"
    ext="${base##*.}"
    digest="$(printf '%s' "$src" | sha256sum | cut -c1-10)"
    if [[ "$stem" == "$base" ]]; then
      dest="$dest_dir/${base}_${digest}"
    else
      dest="$dest_dir/${stem}_${digest}.${ext}"
    fi
  fi

  cp -aL "$src" "$dest"
}

copy_compact_tree() {
  local src="$1"
  local dest="$2"
  [[ -d "$src" ]] || return 0

  mkdir -p "$dest"

  while IFS= read -r -d '' file; do
    local rel target
    rel="${file#"$src"/}"
    target="$dest/$rel"
    mkdir -p "$(dirname "$target")"
    cp -aL "$file" "$target"
  done < <(
    find "$src" -type f \
      ! -name '*.pt' \
      ! -name '*.cache' \
      ! -name '*.npy' \
      ! -name '*.npz' \
      ! -name '*.tmp' \
      ! -path '*/__pycache__/*' \
      -print0
  )
}

copy_matching_models() {
  local src="$1"
  local dest="$2"
  shift 2
  [[ -d "$src" ]] || return 0

  mkdir -p "$dest"

  local pattern file rel target
  for pattern in "$@"; do
    while IFS= read -r -d '' file; do
      rel="${file#"$src"/}"
      target="$dest/$rel"
      mkdir -p "$(dirname "$target")"
      cp -aL "$file" "$target"
    done < <(find "$src" -type f -name "$pattern" -print0)
  done
}

copy_all_pt_models() {
  local src="$1"
  local dest="$2"
  [[ -d "$src" ]] || return 0

  mkdir -p "$dest"

  while IFS= read -r -d '' file; do
    local rel target
    rel="${file#"$src"/}"
    target="$dest/$rel"
    mkdir -p "$(dirname "$target")"
    cp -aL "$file" "$target"
  done < <(find "$src" -type f -name '*.pt' -print0)
}

stage1_dir="$(resolve_run_dir \
  "$RESULTS_ROOT/stage1_results" \
  "$RESULTS_ROOT/stage1_results")"

stage2_dir="$(resolve_run_dir \
  "$STAGE2_PREFERRED" \
  "$RESULTS_ROOT/stage2_results")"

stage3_dir="$(resolve_run_dir \
  "$STAGE3_PREFERRED" \
  "$RESULTS_ROOT/stage3_results")"

stage4_dir="$(resolve_run_dir \
  "$STAGE4_PREFERRED" \
  "$RESULTS_ROOT/stage4_results")"

if [[ -L "$RESULTS_ROOT/stage4_results/latest" ]]; then
  stage4_latest="$(readlink -f "$RESULTS_ROOT/stage4_results/latest")"
  if [[ -d "$stage4_latest" ]]; then
    stage4_dir="$stage4_latest"
  fi
fi

log "Stage 1 source: ${stage1_dir:-not found}"
log "Stage 2 source: ${stage2_dir:-not found}"
log "Stage 3 source: ${stage3_dir:-not found}"
log "Stage 4 source: ${stage4_dir:-not found}"

# ---------------------------------------------------------------------------
# Stage 1: compact baseline/environment files only. Never copy datasets.
# ---------------------------------------------------------------------------
if [[ -n "${stage1_dir:-}" && -d "$stage1_dir" ]]; then
  copy_compact_tree \
    "$stage1_dir" \
    "$work_dir/01_stage1_baselines_and_environment/$(basename "$stage1_dir")"
fi

# Include the original reproduction bundle when it exists.
if [[ "$INCLUDE_REPRODUCTION_ZIP" == "1" ]]; then
  for candidate in \
    "$PROJECT_ROOT/reproducing_files_for_hoyin.zip" \
    "$PROJECT_ROOT/0_inputs/reproducing_files_for_hoyin.zip" \
    "$RESULTS_ROOT/reproducing_files_for_hoyin.zip"
  do
    if [[ -f "$candidate" ]]; then
      copy_file_preserve_name \
        "$candidate" \
        "$work_dir/01_stage1_baselines_and_environment"
      break
    fi
  done
fi

# ---------------------------------------------------------------------------
# Stage 2: all layer result tables/settings, selected L9 models by default.
# ---------------------------------------------------------------------------
if [[ -n "${stage2_dir:-}" && -d "$stage2_dir" ]]; then
  stage2_dest="$work_dir/02_stage2_layer_sweep/$(basename "$stage2_dir")"
  copy_compact_tree "$stage2_dir" "$stage2_dest"

  if [[ "$INCLUDE_ALL_STAGE2_MODELS" == "1" ]]; then
    log "Including every Stage 2 checkpoint."
    copy_all_pt_models "$stage2_dir" "$stage2_dest"
  else
    log "Including selected Stage 2 L9 checkpoints."
    copy_matching_models \
      "$stage2_dir" \
      "$stage2_dest" \
      'GEN_replace_L9_*.pt' \
      'SNOW_replace_L9_*.pt' \
      '*L9*SPPF*.pt' \
      '*layer_9*.pt'
  fi
fi

# ---------------------------------------------------------------------------
# Stage 3: all candidate result tables/settings, selected C007 models by default.
# ---------------------------------------------------------------------------
if [[ -n "${stage3_dir:-}" && -d "$stage3_dir" ]]; then
  stage3_dest="$work_dir/03_stage3_candidate_search/$(basename "$stage3_dir")"
  copy_compact_tree "$stage3_dir" "$stage3_dest"

  if [[ "$INCLUDE_ALL_STAGE3_MODELS" == "1" ]]; then
    log "Including every Stage 3 checkpoint."
    copy_all_pt_models "$stage3_dir" "$stage3_dest"
  else
    log "Including selected Stage 3 C007 checkpoints."
    copy_matching_models \
      "$stage3_dir" \
      "$stage3_dest" \
      '*C007*9-19*.pt' \
      '*C007*layers_9-19*.pt' \
      '*C007*.pt'
  fi
fi

# ---------------------------------------------------------------------------
# Stage 4: include the complete successful recovery folder.
# This contains summaries, comparison tables, validation outputs, and all
# C007/L9 recovered best.pt and last.pt checkpoints for GEN and SNOW.
# ---------------------------------------------------------------------------
if [[ -n "${stage4_dir:-}" && -d "$stage4_dir" ]]; then
  log "Copying complete Stage 4 recovery folder."
  cp -aL \
    "$stage4_dir" \
    "$work_dir/04_stage4_recovery/$(basename "$stage4_dir")"
else
  echo "ERROR: A Stage 4 recovery folder was not found." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Experiment scripts from Stages 1-4.
# ---------------------------------------------------------------------------
for script_dir in \
  "$SCRIPTS_ROOT/layer_replacement_reproduction_stage1" \
  "$SCRIPTS_ROOT/layer_replacement_reproduction_stage2" \
  "$SCRIPTS_ROOT/layer_replacement_reproduction_stage3" \
  "$SCRIPTS_ROOT/layer_replacement_reproduction_stage4"
do
  if [[ -d "$script_dir" ]]; then
    cp -aL "$script_dir" "$work_dir/05_experiment_scripts/"
  fi
done

# Also include files matching the main stage script naming pattern when the
# folders have different names.
while IFS= read -r -d '' file; do
  copy_file_preserve_name "$file" "$work_dir/05_experiment_scripts/other_stage_files"
done < <(
  find "$SCRIPTS_ROOT" -maxdepth 4 -type f \
    \( -name '01_*.py' -o -name '02_*.py' -o -name '03_*.py' \
       -o -name '04_*.py' -o -name '05_*.py' \
       -o -name 'run_stage*.sh' \
       -o -name 'layer_replacement_adapter.py' \) \
    -print0 2>/dev/null
)

# ---------------------------------------------------------------------------
# Copy referenced checkpoint inputs listed inside Stage 4 CSV/JSON files.
# This normally captures baseline and raw source checkpoints.
# ---------------------------------------------------------------------------
export STAGE4_SOURCE_DIR="$stage4_dir"
export REFERENCED_MODEL_DEST="$work_dir/06_selected_and_referenced_models"

python3 - <<'PY'
import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

source = Path(os.environ["STAGE4_SOURCE_DIR"])
dest = Path(os.environ["REFERENCED_MODEL_DEST"])
dest.mkdir(parents=True, exist_ok=True)

references: set[Path] = set()
path_pattern = re.compile(r"(/[^\s\"',]+?\.pt)\b")

def add_value(value):
    if not isinstance(value, str):
        return
    for match in path_pattern.findall(value):
        p = Path(match)
        if p.is_file():
            references.add(p.resolve())

for csv_path in source.rglob("*.csv"):
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for value in row.values():
                    add_value(value)
    except Exception:
        pass

for json_path in source.rglob("*.json"):
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        continue

    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        else:
            add_value(item)

copied = []
for src in sorted(references):
    base = src.name
    out = dest / base

    if out.exists():
        tag = hashlib.sha256(str(src).encode()).hexdigest()[:10]
        out = dest / f"{src.stem}_{tag}{src.suffix}"

    shutil.copy2(src, out)
    copied.append((str(src), str(out.relative_to(dest.parent))))

manifest = dest.parent / "referenced_model_sources.tsv"
with manifest.open("w", encoding="utf-8") as f:
    f.write("source_path\tpackaged_path\n")
    for src, out in copied:
        f.write(f"{src}\t{out}\n")

print(f"Copied {len(copied)} referenced checkpoint(s).")
PY

# ---------------------------------------------------------------------------
# Package notes and inventory.
# ---------------------------------------------------------------------------
cat > "$work_dir/00_package_information/README.txt" <<EOF
YOLO26n Stage 1-4 handoff package
Generated: $(date --iso-8601=seconds)

Purpose
-------
This package collects the useful experiment evidence needed to begin Stage 5.

Included
--------
1. Stage 1 compact baseline/environment outputs, when found.
2. Stage 2 complete layer-sweep tables, settings, and non-checkpoint outputs.
3. Stage 2 selected L9 raw GEN/SNOW checkpoints.
4. Stage 3 complete near-target candidate tables, settings, and non-checkpoint outputs.
5. Stage 3 selected C007 raw GEN/SNOW checkpoints.
6. Complete Stage 4 successful C007/L9 recovery output:
   - stage4_progress.csv
   - stage4_summary.json
   - baseline_raw_recovered_comparison.csv
   - source inventory and checksums
   - architecture audit files
   - training and final-validation outputs
   - recovered best.pt and last.pt models for GEN and SNOW
7. Experiment scripts from Stages 1-4.
8. Referenced checkpoint inputs discovered in Stage 4 CSV/JSON records.
9. SHA256SUMS and MANIFEST.tsv.

Excluded
--------
- Raw GEN and SNOW datasets
- Dataset image/label copies
- Cache files
- Temporary files
- All Stage 2 and Stage 3 checkpoints unless explicitly requested

Source runs
-----------
Stage 1: ${stage1_dir:-not found}
Stage 2: ${stage2_dir:-not found}
Stage 3: ${stage3_dir:-not found}
Stage 4: ${stage4_dir:-not found}

Packaging options
-----------------
INCLUDE_ALL_STAGE2_MODELS=$INCLUDE_ALL_STAGE2_MODELS
INCLUDE_ALL_STAGE3_MODELS=$INCLUDE_ALL_STAGE3_MODELS
INCLUDE_REPRODUCTION_ZIP=$INCLUDE_REPRODUCTION_ZIP
EOF

python3 - <<PY
from pathlib import Path
import hashlib

root = Path(r"$work_dir")
manifest = root / "00_package_information" / "MANIFEST.tsv"
checksums = root / "00_package_information" / "SHA256SUMS"

files = sorted(
    p for p in root.rglob("*")
    if p.is_file() and p not in {manifest, checksums}
)

with manifest.open("w", encoding="utf-8") as f:
    f.write("relative_path\tsize_bytes\n")
    for p in files:
        f.write(f"{p.relative_to(root)}\t{p.stat().st_size}\n")

with checksums.open("w", encoding="utf-8") as f:
    for p in files:
        h = hashlib.sha256()
        with p.open("rb") as src:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                h.update(chunk)
        f.write(f"{h.hexdigest()}  {p.relative_to(root)}\n")

print(f"Manifest entries: {len(files)}")
PY

# ---------------------------------------------------------------------------
# Create ZIP using Python so the command does not depend on the system zip tool.
# ---------------------------------------------------------------------------
log "Creating ZIP archive."

export PACKAGE_WORK_DIR="$work_dir"
export PACKAGE_ZIP_PATH="$zip_path"

python3 - <<'PY'
import os
import zipfile
from pathlib import Path

root = Path(os.environ["PACKAGE_WORK_DIR"])
zip_path = Path(os.environ["PACKAGE_ZIP_PATH"])

with zipfile.ZipFile(
    zip_path,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6,
    allowZip64=True,
) as zf:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            arcname = Path(root.name) / path.relative_to(root)
            zf.write(path, arcname)

print(zip_path)
PY

zip_sha="$(sha256sum "$zip_path" | awk '{print $1}')"
zip_size="$(du -h "$zip_path" | awk '{print $1}')"

cat > "${zip_path}.sha256" <<EOF
${zip_sha}  $(basename "$zip_path")
EOF

echo
echo "===== PACKAGE COMPLETE ====="
echo "ZIP:       $zip_path"
echo "Size:      $zip_size"
echo "SHA-256:   $zip_sha"
echo "Checksum:  ${zip_path}.sha256"
echo
echo "The uncompressed staging folder remains at:"
echo "$work_dir"
